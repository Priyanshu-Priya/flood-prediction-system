"""
Pan-India Training Driver: Multi-Station FloodLSTM (Normalized)
==============================================================
Aggregates seeded GloFAS data, performs normalization, and 
executes a unified training run with numerical stability.
"""

import sys
import os
import pandas as pd
import numpy as np
import xarray as xr
import torch
import joblib
from pathlib import Path
from loguru import logger
from torch.utils.data import ConcatDataset, random_split
from sklearn.preprocessing import StandardScaler

# Add project root to path
sys.path.insert(0, os.getcwd())

from config.settings import settings, RAW_DATA_DIR, MODEL_DIR
from src.models.lstm_forecaster import FloodDataset, FloodLSTM
from src.models.training.train_lstm import LSTMTrainer
from src.ingestion.glofas import INDIA_GAUGE_STATIONS

def compute_api(series: np.ndarray, window: int, decay: float = 0.9) -> np.ndarray:
    """Compute Antecedent Precipitation Index (proxy using discharge)."""
    weights = np.power(decay, np.arange(window)[::-1])
    api = np.convolve(series, weights, mode='full')[:len(series)]
    return api

def prepare_station_data(lat: float, lon: float):
    """Loads and feature-engineers data for a specific station pixel."""
    pattern = f"glofas_2019_*_{lat:.2f}_{lon:.2f}.nc"
    files = sorted(list((RAW_DATA_DIR / "glofas").glob(pattern)))
    
    if not files:
        return None, None, None

    datasets = [xr.open_dataset(f) for f in files]
    ds = xr.concat(datasets, dim="time").sortby("time")
    discharge = ds["dis24"].isel(latitude=0, longitude=0).values
    
    valid_idx = ~np.isnan(discharge)
    discharge = discharge[valid_idx]
    
    if len(discharge) < settings.lstm.lookback_steps + settings.lstm.forecast_steps:
        return None, None, None

    n = len(discharge)
    features = np.zeros((n, 12), dtype=np.float32)
    features[:, 0] = discharge
    features[:, 1] = np.log1p(np.maximum(0, discharge))
    features[:, 2] = np.gradient(discharge)
    features[:, 3] = compute_api(discharge, 3)
    features[:, 4] = compute_api(discharge, 7)
    features[:, 5] = compute_api(discharge, 14)
    features[:, 6] = compute_api(discharge, 30)
    
    df_temp = pd.Series(discharge)
    features[:, 7] = df_temp.rolling(7, min_periods=1).mean().values
    features[:, 8] = df_temp.rolling(7, min_periods=1).std().fillna(0).values
    
    doy = pd.to_datetime(ds.time.values[valid_idx]).dayofyear
    features[:, 9] = np.sin(2 * np.pi * doy / 365.25)
    features[:, 10] = np.cos(2 * np.pi * doy / 365.25)
    features[:, 11] = discharge / (np.max(discharge) + 1e-6)
    
    static = np.zeros(8, dtype=np.float32)
    static[0] = lat
    static[1] = lon
    
    return features, static, discharge

def main():
    logger.info("Step 1: Discovering Seeded Station Data...")
    nc_files = list((RAW_DATA_DIR / "glofas").glob("*.nc"))
    coords = set()
    for f in nc_files:
        parts = f.stem.split('_')
        if len(parts) >= 5:
            coords.add((float(parts[3]), float(parts[4])))
    
    logger.info(f"Found {len(coords)} station coordinates.")
    
    all_stations_data = [] # List of (dyn, stat, target) tuples
    
    logger.info("Step 2: Performing Feature Engineering & Normalization...")
    for lat, lon in sorted(coords):
        dyn, stat, target = prepare_station_data(lat, lon)
        if dyn is not None:
            all_stations_data.append((dyn, stat, target))
    
    if not all_stations_data:
        logger.error("No valid datasets loaded.")
        return

    # ── Unified Normalization ──
    # Concatenate all to fit scalers
    full_dynamic = np.concatenate([d[0] for d in all_stations_data], axis=0)
    full_target = np.concatenate([d[2] for d in all_stations_data], axis=0)
    
    # Exclude DOY sine/cosine from normalization if desired, but we'll scale all for simplicity
    dyn_scaler = StandardScaler()
    target_scaler = StandardScaler()
    
    dyn_scaler.fit(full_dynamic)
    target_scaler.fit(full_target.reshape(-1, 1))
    
    # Save scalers for inference
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(dyn_scaler, MODEL_DIR / "feature_scaler.joblib")
    joblib.dump(target_scaler, MODEL_DIR / "target_scaler.joblib")
    logger.success("Scalers fitted and saved to models/checkpoints/")

    # ── Create Dataset Pool ──
    all_datasets = []
    for dyn, stat, target in all_stations_data:
        # Scale
        dyn_scaled = dyn_scaler.transform(dyn)
        target_scaled = target_scaler.transform(target.reshape(-1, 1)).flatten()
        
        ds = FloodDataset(
            dyn_scaled, stat, target_scaled,
            lookback=settings.lstm.lookback_steps,
            forecast_horizon=settings.lstm.forecast_steps
        )
        all_datasets.append(ds)

    full_dataset = ConcatDataset(all_datasets)
    logger.success(f"Step 3: Unified Dataset ready | total_samples={len(full_dataset)}")

    # Split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = random_split(
        full_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )

    logger.info(f"Step 4: Mixed-Station Split | train={train_size} | val={val_size}")

    trainer = LSTMTrainer(
        n_dynamic_features=12,
        n_static_features=8,
        device=settings.lstm.device
    )

    logger.info("Step 5: Starting LSTM Training Loop (with Normalization)...")
    results = trainer.train(
        train_subset, val_subset,
        output_dir=MODEL_DIR
    )

    logger.success("Step 6: Training Finished!")
    logger.info(f"Final Validation NSE: {results['best_val_nse']:.4f}")

if __name__ == "__main__":
    main()
