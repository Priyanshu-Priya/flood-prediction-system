"""
XGBoost Spatial Flood Susceptibility Mapping
=============================================
Binary classification: flood / no-flood per grid cell.

Uses terrain, LULC, and dynamic weather features to produce
a spatially explicit flood probability map.

Why XGBoost (not a CNN)?
- Tabular features (TWI, slope, etc.) are naturally suited for tree models
- Handles mixed feature types (continuous + categorical) natively
- GPU-accelerated on RTX 4050 via tree_method="gpu_hist"
- Excellent interpretability via SHAP feature importance
- More sample-efficient than deep learning for structured data

The spatial model answers: "Given these terrain and weather conditions,
how likely is this specific location to flood?"
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import StratifiedKFold

from config.settings import settings, MODEL_DIR


class SpatialFloodSusceptibility:
    """
    XGBoost-based spatial flood susceptibility classifier.

    Feature set per grid cell:
    ┌──────────────────┬───────────────────────┬──────────┐
    │ Feature          │ Source                │ Type     │
    ├──────────────────┼───────────────────────┼──────────┤
    │ TWI              │ DEM                   │ Static   │
    │ Slope            │ DEM                   │ Static   │
    │ Aspect           │ DEM                   │ Static   │
    │ Elevation        │ DEM                   │ Static   │
    │ Flow Accum.      │ DEM                   │ Static   │
    │ Dist to Channel  │ DEM + drainage        │ Static   │
    │ Curvature        │ DEM                   │ Static   │
    │ LULC class       │ ESA WorldCover        │ Semi-st  │
    │ Impervious %     │ WorldCover            │ Semi-st  │
    │ Runoff coeff.    │ WorldCover            │ Semi-st  │
    │ API (7d, 14d)    │ IMD / GPM             │ Dynamic  │
    │ Soil moisture    │ SMAP / ERA5           │ Dynamic  │
    │ SAR water freq.  │ Sentinel-1            │ Dynamic  │
    │ Hist. flood count│ Records / SAR archive │ Static   │
    └──────────────────┴───────────────────────┴──────────┘

    Target: Binary flood label (from SAR flood masks or inventory)
    """

    FEATURE_NAMES = [
        "twi", "slope", "aspect", "elevation", "flow_accumulation",
        "distance_to_channel", "curvature", "lulc_class",
        "impervious_fraction", "runoff_coefficient",
        "api_7d", "api_14d", "soil_moisture",
        "sar_water_frequency", "historical_flood_count",
    ]

    def __init__(self):
        self.model = None
        self.feature_importance = None
        self.best_params = None

    def train(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray,
        eval_set: Optional[tuple] = None,
        optimize_hyperparams: bool = True,
    ) -> dict:
        """
        Train XGBoost flood susceptibility model.

        Args:
            X: Feature matrix (n_cells, n_features)
            y: Binary labels (1=flood, 0=no-flood)
            eval_set: Optional (X_val, y_val) for early stopping
            optimize_hyperparams: If True, run Optuna HPO first

        Returns:
            Training results dictionary
        """
        import xgboost as xgb

        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
        else:
            feature_names = self.FEATURE_NAMES[:X.shape[1]]

        logger.info(
            f"Training XGBoost | samples={len(y)} | "
            f"features={X.shape[1]} | pos_rate={y.mean():.4f}"
        )

        # Handle class imbalance
        scale_pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)

        if optimize_hyperparams:
            self.best_params = self._optimize_hyperparams(X, y)
        else:
            self.best_params = self._default_params()

        self.best_params["scale_pos_weight"] = scale_pos_weight

        # Train final model
        dtrain = xgb.DMatrix(X, label=y, feature_names=feature_names)

        if eval_set:
            dval = xgb.DMatrix(eval_set[0], label=eval_set[1], feature_names=feature_names)
            watchlist = [(dtrain, "train"), (dval, "val")]
        else:
            watchlist = [(dtrain, "train")]

        self.model = xgb.train(
            self.best_params,
            dtrain,
            num_boost_round=settings.xgboost.n_estimators,
            evals=watchlist,
            early_stopping_rounds=settings.xgboost.early_stopping_rounds,
            verbose_eval=50,
        )

        # Feature importance
        importance = self.model.get_score(importance_type="gain")
        self.feature_importance = pd.Series(importance).sort_values(ascending=False)

        logger.info(f"Top 5 features:\n{self.feature_importance.head()}")

        return {
            "best_iteration": self.model.best_iteration,
            "best_score": self.model.best_score,
            "feature_importance": self.feature_importance.to_dict(),
        }

    def predict_probability(
        self,
        X: pd.DataFrame | np.ndarray,
    ) -> np.ndarray:
        """
        Predict flood probability for each grid cell.

        Args:
            X: Feature matrix (n_cells, n_features)

        Returns:
            Probability array (0-1) for each cell
        """
        import xgboost as xgb

        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
        else:
            feature_names = self.FEATURE_NAMES[:X.shape[1]]

        dmatrix = xgb.DMatrix(X, feature_names=feature_names)
        probabilities = self.model.predict(dmatrix)

        logger.info(
            f"Predicted flood probabilities | "
            f"mean={probabilities.mean():.4f} | "
            f"max={probabilities.max():.4f} | "
            f"p>0.5: {(probabilities > 0.5).mean():.4f}"
        )

        return probabilities

    def predict_spatial_map(
        self,
        feature_stack: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generate 2D flood probability map from stacked feature rasters.

        Args:
            feature_stack: (n_features, height, width) — stacked feature bands
            mask: (height, width) boolean — valid data mask

        Returns:
            (height, width) flood probability map
        """
        n_features, height, width = feature_stack.shape

        # Reshape to tabular: (n_pixels, n_features)
        pixels = feature_stack.reshape(n_features, -1).T  # (H*W, n_features)

        # Apply mask if provided
        if mask is not None:
            valid_pixels = mask.ravel()
        else:
            valid_pixels = ~np.isnan(pixels).any(axis=1)

        # Fill NaN for valid pixels (XGBoost handles NaN natively)
        result = np.full(height * width, np.nan, dtype=np.float32)
        result[valid_pixels] = self.predict_probability(pixels[valid_pixels])

        return result.reshape(height, width)

    def spatial_cross_validation(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        watershed_ids: np.ndarray,
        n_folds: int = 5,
    ) -> dict:
        """
        Spatial k-fold cross-validation (leave-one-watershed-out).

        Standard k-fold would leak spatial autocorrelation:
        adjacent pixels in training and validation sets have correlated
        features, leading to overly optimistic performance estimates.

        Spatial CV ensures entire watersheds are held out,
        giving honest generalization estimates.

        Args:
            X: Feature DataFrame
            y: Binary labels
            watershed_ids: Array mapping each sample to its watershed
            n_folds: Number of spatial folds

        Returns:
            CV results with per-fold metrics
        """
        from sklearn.metrics import roc_auc_score, brier_score_loss

        unique_watersheds = np.unique(watershed_ids)
        n_watersheds = len(unique_watersheds)

        logger.info(
            f"Spatial CV | {n_watersheds} watersheds | {n_folds} folds"
        )

        # Assign watersheds to folds
        np.random.shuffle(unique_watersheds)
        fold_assignments = np.array_split(unique_watersheds, n_folds)

        fold_results = []

        for fold_idx, val_watersheds in enumerate(fold_assignments):
            val_mask = np.isin(watershed_ids, val_watersheds)
            train_mask = ~val_mask

            X_train, X_val = X[train_mask], X[val_mask]
            y_train, y_val = y[train_mask], y[val_mask]

            logger.info(
                f"Fold {fold_idx + 1}/{n_folds} | "
                f"train={train_mask.sum()} | val={val_mask.sum()} | "
                f"val_watersheds={len(val_watersheds)}"
            )

            # Train on this fold
            self.train(X_train, y_train, eval_set=(X_val, y_val), optimize_hyperparams=False)

            # Evaluate
            y_prob = self.predict_probability(X_val)

            auc = roc_auc_score(y_val, y_prob)
            brier = brier_score_loss(y_val, y_prob)

            fold_results.append({
                "fold": fold_idx + 1,
                "n_val_watersheds": len(val_watersheds),
                "n_val_samples": val_mask.sum(),
                "auc_roc": auc,
                "brier_score": brier,
            })

            logger.info(f"Fold {fold_idx + 1} | AUC={auc:.4f} | Brier={brier:.4f}")

        results_df = pd.DataFrame(fold_results)
        logger.info(
            f"\nSpatial CV Summary:\n"
            f"  Mean AUC-ROC: {results_df['auc_roc'].mean():.4f} ± "
            f"{results_df['auc_roc'].std():.4f}\n"
            f"  Mean Brier:   {results_df['brier_score'].mean():.4f} ± "
            f"{results_df['brier_score'].std():.4f}"
        )

        return {
            "fold_results": results_df,
            "mean_auc": results_df["auc_roc"].mean(),
            "std_auc": results_df["auc_roc"].std(),
            "mean_brier": results_df["brier_score"].mean(),
        }

    def save_model(self, path: Optional[Path] = None) -> Path:
        """Save trained model to file."""
        if path is None:
            path = MODEL_DIR / "xgboost_spatial_susceptibility.json"

        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))
        logger.info(f"XGBoost model saved: {path}")
        return path

    def load_model(self, path: Path) -> None:
        """Load trained model from file."""
        import xgboost as xgb

        self.model = xgb.Booster()
        self.model.load_model(str(path))
        logger.info(f"XGBoost model loaded: {path}")

    def _default_params(self) -> dict:
        """Default XGBoost parameters optimized for flood susceptibility."""
        s = settings.xgboost
        return {
            "objective": "binary:logistic",
            "eval_metric": s.eval_metric,
            "max_depth": s.max_depth,
            "learning_rate": s.learning_rate,
            "subsample": s.subsample,
            "colsample_bytree": s.colsample_bytree,
            "min_child_weight": s.min_child_weight,
            "gamma": s.gamma,
            "reg_alpha": s.reg_alpha,
            "reg_lambda": s.reg_lambda,
            "tree_method": s.tree_method,
            "device": s.device,
            "seed": 42,
        }

    def _optimize_hyperparams(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray,
        n_trials: int = 50,
    ) -> dict:
        """
        Optimize hyperparameters using Optuna Bayesian search.

        Optimizes: max_depth, learning_rate, subsample, colsample_bytree,
        min_child_weight, gamma, reg_alpha, reg_lambda
        """
        import optuna
        import xgboost as xgb

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = {
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0, 5),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
                "tree_method": settings.xgboost.tree_method,
                "device": settings.xgboost.device,
                "seed": 42,
            }

            dtrain = xgb.DMatrix(X, label=y)
            cv_results = xgb.cv(
                params,
                dtrain,
                num_boost_round=500,
                nfold=3,
                stratified=True,
                early_stopping_rounds=30,
                verbose_eval=False,
            )

            return cv_results["test-auc-mean"].iloc[-1]

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        logger.info(
            f"Optuna HPO complete | best AUC={study.best_value:.4f} | "
            f"best_params={study.best_params}"
        )

        best = study.best_params
        best.update({
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": settings.xgboost.tree_method,
            "device": settings.xgboost.device,
            "seed": 42,
        })

        return best
