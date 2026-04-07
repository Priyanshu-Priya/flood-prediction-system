import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
import pandas as pd
from loguru import logger

from src.models.training.train_lstm import LSTMTrainer
from src.models.lstm_forecaster import FloodDataset
from config.settings import settings, MODEL_DIR
from src.models.spatial_susceptibility import SpatialFloodSusceptibility

def train_lstm():
    logger.info("Initializing synthetic dataset for LSTM...")
    N_SAMPLES = 2000
    n_dynamic = 12
    n_static = 8
    
    # 1. Create a synthetic wave representing "water levels"
    t = np.linspace(0, 10 * np.pi, N_SAMPLES)
    water_levels = np.sin(t) * 2 + 10 
    
    # 2. Build feature tensors expected by the model
    dynamic_features = np.random.randn(N_SAMPLES, n_dynamic).astype(np.float32)
    dynamic_features[:, 0] = water_levels 
    static_features = np.random.randn(n_static).astype(np.float32)
    targets = water_levels.astype(np.float32)

    # 3. Create Datasets for LSTM sliding window processing
    train_dataset = FloodDataset(
        dynamic_features=dynamic_features[:1500],
        static_features=static_features,
        targets=targets[:1500],
        lookback=settings.lstm.lookback_steps,
        forecast_horizon=settings.lstm.forecast_steps
    )
    val_dataset = FloodDataset(
        dynamic_features=dynamic_features[1500:],
        static_features=static_features,
        targets=targets[1500:],
        lookback=settings.lstm.lookback_steps,
        forecast_horizon=settings.lstm.forecast_steps
    )

    # Fast forward: 1 epoch is enough to initialize weights and save the checkpoint
    settings.lstm.max_epochs = 1
    
    trainer = LSTMTrainer(
        n_dynamic_features=n_dynamic,
        n_static_features=n_static,
    )
    logger.info("Training LSTM (1 epoch) on RTX 4050...")
    trainer.train(train_dataset, val_dataset, output_dir=MODEL_DIR)

def train_xgboost():
    logger.info("Initializing synthetic dataset for XGBoost...")
    X = pd.DataFrame(np.random.randn(1000, 10), columns=[f"feature_{i}" for i in range(10)])
    y = np.random.randint(0, 2, 1000)
    
    model = SpatialFloodSusceptibility()
    logger.info("Training XGBoost...")
    model.train(X, y)
    
    save_path = MODEL_DIR / "xgboost_spatial_susceptibility.json"
    model.save_model(save_path)
    logger.info(f"XGBoost saved to {save_path}")

if __name__ == "__main__":
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    train_lstm()
    train_xgboost()
    logger.info("Both models have been correctly initialized from synthetic data.")
