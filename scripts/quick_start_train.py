"""
Quick Start Training Script
================================
Trains both models (LSTM + XGBoost) using real or synthetic data.

Priority order:
  1. Real data from data/processed/ (from ingest + prepare scripts)
  2. Synthetic fallback if real data not available

Usage:
    python scripts/quick_start_train.py                    # Auto-detect data
    python scripts/quick_start_train.py --epochs 50        # Custom epochs
    python scripts/quick_start_train.py --synthetic-only    # Force synthetic
"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
import pandas as pd
from loguru import logger

from src.models.training.train_lstm import LSTMTrainer
from src.models.lstm_forecaster import FloodDataset
from config.settings import settings, MODEL_DIR, PROCESSED_DATA_DIR
from src.models.spatial_susceptibility import SpatialFloodSusceptibility
from src.evaluation.metrics import HydrologicalMetrics

import click


def load_real_lstm_data() -> tuple | None:
    """Attempt to load real LSTM datasets from data/processed/."""
    dataset_dir = PROCESSED_DATA_DIR / "lstm_datasets"
    train_path = dataset_dir / "train_dataset.npz"
    val_path = dataset_dir / "val_dataset.npz"
    meta_path = dataset_dir / "dataset_metadata.json"

    if not train_path.exists() or not val_path.exists():
        return None

    logger.info(f"Loading real LSTM data from {dataset_dir}")

    train = np.load(train_path)
    val = np.load(val_path)

    with open(meta_path) as f:
        metadata = json.load(f)

    logger.info(
        f"  Train: {train['dynamic'].shape} | Val: {val['dynamic'].shape} | "
        f"Stations: {metadata.get('stations', 'unknown')}"
    )

    return (
        train["dynamic"], train["static"], train["targets"],
        val["dynamic"], val["static"], val["targets"],
        metadata,
    )


def generate_synthetic_lstm_data() -> tuple:
    """Generate synthetic LSTM data for initial model validation."""
    logger.info("Generating synthetic dataset for LSTM...")
    N_SAMPLES = 2000
    n_dynamic = 12
    n_static = 8

    t = np.linspace(0, 10 * np.pi, N_SAMPLES)
    water_levels = np.sin(t) * 2 + 10

    dynamic_features = np.random.randn(N_SAMPLES, n_dynamic).astype(np.float32)
    dynamic_features[:, 0] = water_levels
    # Add correlation: discharge tracks water level
    dynamic_features[:, 1] = water_levels * 100 + np.random.randn(N_SAMPLES) * 50

    static_features = np.random.randn(n_static).astype(np.float32)
    targets = water_levels.astype(np.float32)

    split = 1500
    metadata = {
        "stations": ["SYNTHETIC"],
        "n_dynamic_features": n_dynamic,
        "n_static_features": n_static,
        "train_samples": split,
        "val_samples": N_SAMPLES - split,
    }

    return (
        dynamic_features[:split], static_features, targets[:split],
        dynamic_features[split:], static_features, targets[split:],
        metadata,
    )


def train_lstm(
    epochs: int = 50,
    use_synthetic: bool = False,
) -> dict:
    """Train the LSTM model on real or synthetic data."""
    logger.info("=" * 60)
    logger.info("LSTM TRAINING")
    logger.info("=" * 60)

    # Load data
    real_data = None if use_synthetic else load_real_lstm_data()

    if real_data:
        train_dyn, train_stat, train_tgt, val_dyn, val_stat, val_tgt, metadata = real_data
        data_source = "real"
    else:
        logger.info("No real data found — using synthetic data")
        train_dyn, train_stat, train_tgt, val_dyn, val_stat, val_tgt, metadata = generate_synthetic_lstm_data()
        data_source = "synthetic"

    n_dynamic = train_dyn.shape[1]
    n_static = len(train_stat) if train_stat.ndim == 1 else train_stat.shape[1]

    # Adjust lookback for available data
    lookback = min(settings.lstm.lookback_steps, len(train_dyn) // 4)
    horizon = min(settings.lstm.forecast_steps, len(train_dyn) // 6)

    # For daily GloFAS data, use practical windows
    if data_source == "real":
        lookback = min(30, len(train_dyn) // 4)
        horizon = min(7, len(train_dyn) // 6)

    logger.info(f"Data source: {data_source}")
    logger.info(f"Lookback: {lookback} | Horizon: {horizon}")
    logger.info(f"Features: {n_dynamic} dynamic + {n_static} static")

    # Create datasets
    train_dataset = FloodDataset(
        dynamic_features=train_dyn,
        static_features=train_stat,
        targets=train_tgt,
        lookback=lookback,
        forecast_horizon=horizon,
    )
    val_dataset = FloodDataset(
        dynamic_features=val_dyn,
        static_features=val_stat,
        targets=val_tgt,
        lookback=lookback,
        forecast_horizon=horizon,
    )

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        logger.error("Dataset too small after windowing. Need more data.")
        return {"status": "FAILED", "error": "insufficient_data"}

    logger.info(f"Train windows: {len(train_dataset)} | Val windows: {len(val_dataset)}")

    # Override epochs
    original_epochs = settings.lstm.max_epochs
    settings.lstm.max_epochs = epochs

    # Adjust forecast steps to match
    original_forecast = settings.lstm.forecast_steps
    settings.lstm.forecast_steps = horizon

    trainer = LSTMTrainer(
        n_dynamic_features=n_dynamic,
        n_static_features=n_static,
    )

    logger.info(f"Training LSTM for {epochs} epochs on {settings.lstm.device}...")
    results = trainer.train(train_dataset, val_dataset, output_dir=MODEL_DIR)

    # Restore settings
    settings.lstm.max_epochs = original_epochs
    settings.lstm.forecast_steps = original_forecast

    results["data_source"] = data_source
    results["n_dynamic_features"] = n_dynamic
    results["n_static_features"] = n_static
    results["lookback"] = lookback
    results["horizon"] = horizon

    logger.info(f"\nLSTM Training Complete!")
    logger.info(f"  Best NSE: {results.get('best_val_nse', 'N/A')}")
    logger.info(f"  Epochs: {results.get('total_epochs', 'N/A')}")

    return results


def load_real_xgboost_data() -> tuple | None:
    """Attempt to load real XGBoost features from processed data."""
    # Search for XGBoost features in any AOI subdirectory
    for aoi_dir in PROCESSED_DATA_DIR.iterdir():
        if aoi_dir.is_dir():
            feature_path = aoi_dir / "xgboost_features.parquet"
            label_path = aoi_dir / "xgboost_labels.npy"

            if feature_path.exists() and label_path.exists():
                X = pd.read_parquet(feature_path)
                y = np.load(label_path)
                logger.info(
                    f"Loaded XGBoost data from {aoi_dir.name} | "
                    f"{len(y)} samples × {X.shape[1]} features"
                )
                return X, y

    return None


def train_xgboost(use_synthetic: bool = False) -> dict:
    """Train XGBoost spatial susceptibility model."""
    logger.info("=" * 60)
    logger.info("XGBOOST TRAINING")
    logger.info("=" * 60)

    # Try real data first
    real_data = None if use_synthetic else load_real_xgboost_data()

    if real_data:
        X, y = real_data
        data_source = "real"
    else:
        logger.info("No real terrain data found — using synthetic")
        feature_names = [
            "slope", "aspect", "twi", "flow_accumulation",
            "distance_to_channel", "curvature", "elevation",
            "runoff_coefficient", "api_14d", "impervious_fraction",
        ]
        X = pd.DataFrame(
            np.random.randn(2000, len(feature_names)),
            columns=feature_names,
        )
        # Make TWI correlated with flood label
        X["twi"] = np.abs(X["twi"]) * 5 + 5
        y = (X["twi"] > np.percentile(X["twi"], 70)).astype(int).values
        data_source = "synthetic"

    logger.info(f"Data source: {data_source}")
    logger.info(f"Samples: {len(y)} | Features: {X.shape[1]} | Flood rate: {y.mean():.4f}")

    model = SpatialFloodSusceptibility()

    # Train with eval set
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info("Training XGBoost...")
    train_results = model.train(X_train, y_train, eval_set=(X_val, y_val))

    # Evaluate
    y_prob = model.predict_probability(X_val)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_val, y_prob)

    # Feature importance
    importance = {}
    if model.model:
        try:
            importance = model.model.get_score(importance_type="gain")
        except Exception:
            pass

    # Save model
    save_path = MODEL_DIR / "xgboost_spatial_susceptibility.json"
    model.save_model(save_path)
    logger.info(f"XGBoost saved to {save_path}")

    results = {
        "auc_roc": float(auc),
        "n_features": int(X.shape[1]),
        "n_train_samples": int(len(y_train)),
        "data_source": data_source,
        "feature_importance": {k: float(v) for k, v in importance.items()},
        "feature_names": list(X.columns),
    }

    logger.info(f"\nXGBoost Training Complete!")
    logger.info(f"  AUC-ROC: {auc:.4f}")
    logger.info(f"  Features: {X.shape[1]}")

    return results


def save_training_metrics(lstm_results: dict, xgb_results: dict):
    """Persist training metrics for the API /predict/metrics endpoint."""
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "lstm": {
            "nse_mean": lstm_results.get("best_val_nse", 0.0),
            "best_val_loss": lstm_results.get("best_val_loss", 0.0),
            "total_epochs": lstm_results.get("total_epochs", 0),
            "data_source": lstm_results.get("data_source", "unknown"),
            "parameters": "7.5M",
            "last_train": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "n_dynamic_features": lstm_results.get("n_dynamic_features", 12),
            "n_static_features": lstm_results.get("n_static_features", 8),
            "lookback": lstm_results.get("lookback", 168),
            "horizon": lstm_results.get("horizon", 72),
        },
        "xgboost": {
            "auc_roc": xgb_results.get("auc_roc", 0.0),
            "n_features": xgb_results.get("n_features", 0),
            "n_train_samples": xgb_results.get("n_train_samples", 0),
            "data_source": xgb_results.get("data_source", "unknown"),
            "feature_importance": xgb_results.get("feature_importance", {}),
            "feature_names": xgb_results.get("feature_names", []),
        },
        "system": {
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "is_gpu_accelerated": torch.cuda.is_available(),
        },
    }

    metrics_path = MODEL_DIR / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Training metrics saved → {metrics_path}")
    return metrics


@click.command()
@click.option("--epochs", default=50, help="LSTM training epochs")
@click.option("--synthetic-only", is_flag=True, help="Force synthetic data")
@click.option("--skip-lstm", is_flag=True, help="Skip LSTM training")
@click.option("--skip-xgboost", is_flag=True, help="Skip XGBoost training")
def main(epochs, synthetic_only, skip_lstm, skip_xgboost):
    """Train all models for the flood prediction system."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    lstm_results = {}
    xgb_results = {}

    if not skip_lstm:
        lstm_results = train_lstm(epochs=epochs, use_synthetic=synthetic_only)

    if not skip_xgboost:
        xgb_results = train_xgboost(use_synthetic=synthetic_only)

    # Persist metrics
    metrics = save_training_metrics(lstm_results, xgb_results)

    logger.info("\n" + "=" * 60)
    logger.info("ALL TRAINING COMPLETE")
    logger.info(f"  LSTM NSE:      {lstm_results.get('best_val_nse', 'SKIPPED')}")
    logger.info(f"  XGBoost AUC:   {xgb_results.get('auc_roc', 'SKIPPED')}")
    logger.info(f"  Data source:   {lstm_results.get('data_source', 'N/A')} (LSTM), "
                f"{xgb_results.get('data_source', 'N/A')} (XGBoost)")
    logger.info(f"  Checkpoints:   {MODEL_DIR}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

