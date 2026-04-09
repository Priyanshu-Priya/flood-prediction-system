"""
LSTM Dataset Preparation
===========================
Transforms raw GloFAS discharge time-series into LSTM-ready datasets.

Builds:
  1. Dynamic feature matrix per station (discharge, water_level, API_3d/7d/14d/30d, ...)
  2. Static feature vector per station (danger_level, warning_level, lat, lon, ...)
  3. Target array (water_level)
  4. Train/val splits using walk-forward temporal split (no leakage)

Output: .npz files ready for FloodDataset

Usage:
    python scripts/prepare_lstm_dataset.py
    python scripts/prepare_lstm_dataset.py --stations GLOFAS_PATNA GLOFAS_DELHI
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.logging_config import setup_logging
from config.settings import PROCESSED_DATA_DIR, MODEL_DIR
from src.features.precipitation import AntecedentPrecipitationIndex
from src.ingestion.glofas import INDIA_GAUGE_STATIONS


def build_dynamic_features(df: pd.DataFrame) -> np.ndarray:
    """
    Build dynamic feature matrix from a station's discharge DataFrame.

    Features (12 total):
      0: water_level_m
      1: discharge_cumecs
      2: discharge_log (log-transformed for stability)
      3: discharge_diff (day-over-day change)
      4: api_3d (Antecedent Precipitation Index, 3-day)
      5: api_7d
      6: api_14d
      7: api_30d
      8: discharge_7d_mean (rolling mean)
      9: discharge_7d_std (rolling std — volatility)
     10: level_to_danger_ratio
     11: seasonal_signal (sine encoding of day-of-year)
    """
    features = pd.DataFrame(index=df.index)

    # Core hydrological signals
    features["water_level_m"] = df["water_level_m"]
    features["discharge_cumecs"] = df["discharge_cumecs"]
    features["discharge_log"] = np.log1p(df["discharge_cumecs"].clip(lower=0))
    features["discharge_diff"] = df["discharge_cumecs"].diff().fillna(0)

    # Antecedent Precipitation Index (using discharge as proxy for upstream rainfall)
    api_calculator = AntecedentPrecipitationIndex(decay_factor=0.90)
    discharge_series = df["discharge_cumecs"].fillna(0).values
    api_df = api_calculator.compute_multi_scale_api(discharge_series, windows=[3, 7, 14, 30])
    features["api_3d"] = api_df["api_3d"].values
    features["api_7d"] = api_df["api_7d"].values
    features["api_14d"] = api_df["api_14d"].values
    features["api_30d"] = api_df["api_30d"].values

    # Rolling statistics
    features["discharge_7d_mean"] = df["discharge_cumecs"].rolling(7, min_periods=1).mean()
    features["discharge_7d_std"] = df["discharge_cumecs"].rolling(7, min_periods=1).std().fillna(0)

    # Danger ratio (how close to danger level)
    danger_level = df.get("danger_level_m", pd.Series(10.0, index=df.index))
    if isinstance(danger_level, (int, float)):
        danger_level = pd.Series(danger_level, index=df.index)
    features["level_to_danger_ratio"] = df["water_level_m"] / danger_level.clip(lower=0.1)

    # Seasonal encoding (captures monsoon periodicity)
    doy = pd.to_datetime(df.index).dayofyear
    features["seasonal_signal"] = np.sin(2 * np.pi * doy / 365.25)

    # Fill any remaining NaN
    features = features.fillna(0)

    return features.values.astype(np.float32)


def build_static_features(station: dict) -> np.ndarray:
    """
    Build static feature vector for a station.

    Features (8 total):
      0: latitude
      1: longitude
      2: danger_level_m
      3: warning_level_m
      4: danger_to_warning_gap
      5: lat_normalized (0-1 range over India)
      6: lon_normalized (0-1 range over India)
      7: is_major_basin (Ganga/Brahmaputra = 1, others = 0)
    """
    lat = station.get("lat", 0)
    lon = station.get("lon", 0)
    danger = station.get("danger_level_m", 10.0)
    warning = station.get("warning_level_m", 8.0)

    major_basins = {"Ganga", "Brahmaputra"}
    is_major = 1.0 if station.get("basin", "") in major_basins else 0.0

    static = np.array([
        lat,
        lon,
        danger,
        warning,
        danger - warning,
        (lat - 6.0) / 30.0,  # Normalize over India lat range
        (lon - 68.0) / 30.0,  # Normalize over India lon range
        is_major,
    ], dtype=np.float32)

    return static


def normalize_features(
    dynamic: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize dynamic features (zero mean, unit variance)."""
    mean = np.nanmean(dynamic, axis=0)
    std = np.nanstd(dynamic, axis=0)
    std[std < 1e-8] = 1.0  # Prevent division by zero
    normalized = (dynamic - mean) / std
    return normalized, mean, std


@click.command()
@click.option("--stations", multiple=True, default=None,
              help="Station IDs to process (default: all with data)")
@click.option("--val-fraction", default=0.2, help="Fraction of data for validation")
@click.option("--lookback", default=168, help="Lookback window (hourly steps, 168=7 days)")
@click.option("--horizon", default=72, help="Forecast horizon (hourly steps, 72=3 days)")
def main(stations, val_fraction, lookback, horizon):
    """Prepare LSTM datasets from GloFAS discharge data."""
    setup_logging(level="INFO")

    ts_dir = PROCESSED_DATA_DIR / "glofas_timeseries"
    output_dir = PROCESSED_DATA_DIR / "lstm_datasets"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not ts_dir.exists():
        logger.error(
            f"No GloFAS data found at {ts_dir}. "
            "Run 'python scripts/ingest_all_data.py' first."
        )
        raise SystemExit(1)

    # Find available station data
    parquet_files = list(ts_dir.glob("GLOFAS_*_discharge.parquet"))
    if not parquet_files:
        logger.error("No parquet files found. Run ingestion first.")
        raise SystemExit(1)

    logger.info(f"Found {len(parquet_files)} station datasets")

    # Build station metadata lookup
    station_lookup = {s["station_id"]: s for s in INDIA_GAUGE_STATIONS}

    all_train_dynamic = []
    all_train_targets = []
    all_val_dynamic = []
    all_val_targets = []
    all_static = []
    station_names = []

    for pf in parquet_files:
        sid = pf.stem.replace("_discharge", "")

        if stations and sid not in stations:
            continue

        logger.info(f"\n── Processing: {sid} ──")

        # Load data
        df = pd.read_parquet(pf)
        if len(df) < lookback + horizon + 10:
            logger.warning(f"  Skipping {sid}: only {len(df)} observations (need >{lookback + horizon})")
            continue

        # GloFAS is daily — we keep daily resolution for the LSTM
        # (adjust lookback/horizon accordingly for daily data)
        effective_lookback = min(lookback, len(df) // 3)
        effective_horizon = min(horizon, len(df) // 4)

        # For daily data, use practical windows
        daily_lookback = min(30, len(df) // 3)    # 30 days lookback
        daily_horizon = min(7, len(df) // 4)      # 7 days forecast

        # Build features
        station_meta = station_lookup.get(sid, {"lat": 0, "lon": 0, "danger_level_m": 10, "warning_level_m": 8, "basin": ""})
        dynamic = build_dynamic_features(df)
        static = build_static_features(station_meta)

        # Walk-forward split: train on earlier data, validate on later
        n = len(dynamic)
        split_idx = int(n * (1 - val_fraction))

        train_dynamic = dynamic[:split_idx]
        val_dynamic = dynamic[split_idx:]
        train_targets = df["water_level_m"].values[:split_idx].astype(np.float32)
        val_targets = df["water_level_m"].values[split_idx:].astype(np.float32)

        logger.info(
            f"  Samples: {n} total | train={split_idx} | val={n - split_idx} | "
            f"features={dynamic.shape[1]}"
        )

        all_train_dynamic.append(train_dynamic)
        all_train_targets.append(train_targets)
        all_val_dynamic.append(val_dynamic)
        all_val_targets.append(val_targets)
        all_static.append(static)
        station_names.append(sid)

    if not all_train_dynamic:
        logger.error("No valid station data processed!")
        raise SystemExit(1)

    # Concatenate all stations for multi-station training
    train_dynamic = np.concatenate(all_train_dynamic, axis=0)
    train_targets = np.concatenate(all_train_targets, axis=0)
    val_dynamic = np.concatenate(all_val_dynamic, axis=0)
    val_targets = np.concatenate(all_val_targets, axis=0)

    # Use the first station's static features as default (per-station would need
    # the Dataset to handle variable static inputs — simplification for now)
    static_features = all_static[0]

    # Normalize dynamic features
    train_normalized, feat_mean, feat_std = normalize_features(train_dynamic)
    val_normalized = (val_dynamic - feat_mean) / feat_std

    # Save datasets
    np.savez_compressed(
        output_dir / "train_dataset.npz",
        dynamic=train_normalized,
        static=static_features,
        targets=train_targets,
    )
    np.savez_compressed(
        output_dir / "val_dataset.npz",
        dynamic=val_normalized,
        static=static_features,
        targets=val_targets,
    )
    np.savez_compressed(
        output_dir / "scaler.npz",
        mean=feat_mean,
        std=feat_std,
    )

    # Save metadata
    import json
    metadata = {
        "stations": station_names,
        "n_dynamic_features": int(train_dynamic.shape[1]),
        "n_static_features": len(static_features),
        "train_samples": int(len(train_targets)),
        "val_samples": int(len(val_targets)),
        "daily_lookback": 30,
        "daily_horizon": 7,
        "feature_names": [
            "water_level_m", "discharge_cumecs", "discharge_log",
            "discharge_diff", "api_3d", "api_7d", "api_14d", "api_30d",
            "discharge_7d_mean", "discharge_7d_std",
            "level_to_danger_ratio", "seasonal_signal",
        ],
    }
    with open(output_dir / "dataset_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"\n{'=' * 60}")
    logger.info("LSTM Dataset Preparation Complete!")
    logger.info(f"  Train: {train_normalized.shape} | Val: {val_normalized.shape}")
    logger.info(f"  Features: {train_dynamic.shape[1]} dynamic + {len(static_features)} static")
    logger.info(f"  Stations: {station_names}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
