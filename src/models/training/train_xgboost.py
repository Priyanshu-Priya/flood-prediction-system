"""
XGBoost Training Pipeline
===========================
Training orchestrator for the spatial flood susceptibility model.
Handles data preparation, feature stacking from rasters,
label generation from SAR flood masks, and spatial CV evaluation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import rasterio
from loguru import logger

from config.settings import settings, MODEL_DIR, PROCESSED_DATA_DIR
from src.evaluation.metrics import HydrologicalMetrics
from src.models.spatial_susceptibility import SpatialFloodSusceptibility


class XGBoostTrainer:
    """
    End-to-end training pipeline for flood susceptibility mapping.

    Pipeline:
    1. Load stacked terrain features (GeoTIFF bands)
    2. Load dynamic features (API, soil moisture, SAR frequency)
    3. Generate flood labels from SAR-derived water masks
    4. Train XGBoost with spatial CV
    5. Generate and save flood probability map
    """

    def __init__(self):
        self.model = SpatialFloodSusceptibility()

    def prepare_training_data(
        self,
        terrain_stack_path: Path,
        dynamic_features: dict[str, np.ndarray],
        flood_labels: np.ndarray,
        valid_mask: Optional[np.ndarray] = None,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Prepare tabular training data from spatial rasters.

        Args:
            terrain_stack_path: Path to multi-band terrain GeoTIFF
            dynamic_features: Dict of feature_name → 2D numpy array
            flood_labels: Binary 2D array (1=flood, 0=no-flood)
            valid_mask: 2D boolean mask for valid data regions

        Returns:
            (X, y) — feature DataFrame and label vector
        """
        logger.info("Preparing XGBoost training data from spatial rasters")

        # Load terrain features
        with rasterio.open(str(terrain_stack_path)) as src:
            n_bands = src.count
            height, width = src.height, src.width
            terrain_data = src.read()  # (bands, H, W)
            band_names = [src.descriptions[i] or f"terrain_{i}" for i in range(n_bands)]

        logger.info(
            f"Terrain stack: {n_bands} bands × {height}×{width} | "
            f"names={band_names}"
        )

        # Flatten spatial dimensions
        features = {}
        for i, name in enumerate(band_names):
            features[name] = terrain_data[i].ravel()

        # Add dynamic features
        for name, arr in dynamic_features.items():
            if arr.shape != (height, width):
                logger.warning(
                    f"Dynamic feature '{name}' shape {arr.shape} != "
                    f"terrain shape ({height}, {width}). Skipping."
                )
                continue
            features[name] = arr.ravel()

        X = pd.DataFrame(features)
        y = flood_labels.ravel()

        # Apply validity mask
        if valid_mask is not None:
            valid = valid_mask.ravel()
        else:
            # Exclude nodata pixels
            valid = ~X.isna().any(axis=1).values & (y >= 0)

        X = X[valid].reset_index(drop=True)
        y = y[valid]

        logger.info(
            f"Training data prepared | samples={len(y)} | "
            f"features={X.shape[1]} | flood_rate={y.mean():.4f}"
        )

        return X, y

    def train_and_evaluate(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        watershed_ids: Optional[np.ndarray] = None,
        save_model: bool = True,
    ) -> dict:
        """
        Train XGBoost with spatial cross-validation and save model.

        Args:
            X: Feature DataFrame
            y: Binary flood labels
            watershed_ids: Optional watershed assignments for spatial CV
            save_model: Whether to save the final model

        Returns:
            Training and evaluation results
        """
        # Split into train/val
        from sklearn.model_selection import train_test_split

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        logger.info(
            f"Train/Val split | train={len(y_train)} | val={len(y_val)} | "
            f"train_flood_rate={y_train.mean():.4f}"
        )

        # Train
        train_results = self.model.train(
            X_train, y_train,
            eval_set=(X_val, y_val),
            optimize_hyperparams=True,
        )

        # Evaluate on validation set
        y_prob = self.model.predict_probability(X_val)

        from sklearn.metrics import (
            roc_auc_score, classification_report,
            precision_recall_fscore_support, brier_score_loss,
        )

        auc = roc_auc_score(y_val, y_prob)
        brier = brier_score_loss(y_val, y_prob)

        # Find optimal threshold via F1
        best_f1 = 0
        best_thresh = 0.5
        for thresh in np.arange(0.1, 0.9, 0.05):
            y_pred = (y_prob >= thresh).astype(int)
            p, r, f1, _ = precision_recall_fscore_support(
                y_val, y_pred, average="binary", zero_division=0
            )
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh

        y_pred_final = (y_prob >= best_thresh).astype(int)
        report = classification_report(y_val, y_pred_final, output_dict=True)

        results = {
            "auc_roc": auc,
            "brier_score": brier,
            "best_threshold": best_thresh,
            "best_f1": best_f1,
            "classification_report": report,
            "feature_importance": train_results.get("feature_importance", {}),
        }

        logger.info(
            f"\n{'='*50}\n"
            f"XGBoost Evaluation Results:\n"
            f"  AUC-ROC:    {auc:.4f}\n"
            f"  Brier:      {brier:.4f}\n"
            f"  Best F1:    {best_f1:.4f} (threshold={best_thresh:.2f})\n"
            f"  Precision:  {report['1']['precision']:.4f}\n"
            f"  Recall:     {report['1']['recall']:.4f}\n"
            f"{'='*50}"
        )

        # Spatial CV if watershed IDs provided
        if watershed_ids is not None:
            cv_results = self.model.spatial_cross_validation(
                X, y, watershed_ids
            )
            results["spatial_cv"] = cv_results

        # Save model
        if save_model:
            model_path = self.model.save_model()
            results["model_path"] = str(model_path)

        return results

    def generate_susceptibility_map(
        self,
        terrain_stack_path: Path,
        dynamic_features: dict[str, np.ndarray],
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Generate spatial flood susceptibility map using trained model.

        Args:
            terrain_stack_path: Multi-band terrain feature GeoTIFF
            dynamic_features: Current dynamic feature arrays
            output_path: Where to save the probability GeoTIFF

        Returns:
            Path to output flood probability GeoTIFF
        """
        if output_path is None:
            output_path = PROCESSED_DATA_DIR / "flood_susceptibility_map.tif"

        logger.info("Generating flood susceptibility map")

        with rasterio.open(str(terrain_stack_path)) as src:
            terrain_data = src.read()  # (bands, H, W)
            meta = src.meta.copy()

        # Stack all features
        all_features = [terrain_data]
        for name, arr in dynamic_features.items():
            all_features.append(arr[np.newaxis, :, :])

        feature_stack = np.concatenate(all_features, axis=0)

        # Predict
        prob_map = self.model.predict_spatial_map(feature_stack)

        # Save as GeoTIFF
        meta.update({
            "count": 1,
            "dtype": "float32",
            "nodata": -9999,
            "compress": "lzw",
        })

        with rasterio.open(str(output_path), "w", **meta) as dst:
            prob_map_clean = np.nan_to_num(prob_map, nan=-9999)
            dst.write(prob_map_clean.astype(np.float32), 1)
            dst.set_band_description(1, "flood_probability")

        logger.info(f"Susceptibility map saved: {output_path}")
        return output_path
