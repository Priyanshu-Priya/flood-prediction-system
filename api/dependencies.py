"""
FastAPI Dependencies — Model Loading & Shared Resources
=========================================================
Dependency injection for model instances, database connections,
and shared state across API endpoints.
"""

from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path
from typing import Optional

from loguru import logger

from config.settings import settings, MODEL_DIR


_start_time = time.time()


@lru_cache(maxsize=1)
def get_lstm_model():
    """
    Load the LSTM flood forecaster model (cached singleton).

    The model is loaded once at startup and cached in memory.
    Subsequent requests reuse the same instance for 0-latency inference.
    """
    from src.models.lstm_forecaster import FloodForecaster

    checkpoint_path = MODEL_DIR / "best_lstm_model.pt"

    if not checkpoint_path.exists():
        logger.warning(f"LSTM checkpoint not found: {checkpoint_path}")
        return None

    try:
        model = FloodForecaster(str(checkpoint_path))
        logger.info("LSTM model loaded and cached")
        return model
    except Exception as e:
        logger.error(f"Failed to load LSTM model: {e}")
        return None


@lru_cache(maxsize=1)
def get_xgboost_model():
    """Load the XGBoost spatial model (cached singleton)."""
    from src.models.spatial_susceptibility import SpatialFloodSusceptibility

    model_path = MODEL_DIR / "xgboost_spatial_susceptibility.json"

    if not model_path.exists():
        logger.warning(f"XGBoost model not found: {model_path}")
        return None

    try:
        model = SpatialFloodSusceptibility()
        model.load_model(model_path)
        logger.info("XGBoost model loaded and cached")
        return model
    except Exception as e:
        logger.error(f"Failed to load XGBoost model: {e}")
        return None


@lru_cache(maxsize=1)
def get_ensemble_combiner():
    """Get the ensemble combiner instance."""
    from src.models.ensemble import EnsembleCombiner
    return EnsembleCombiner()


def get_gauge_client():
    """Get India-WRIS gauge data client."""
    from src.ingestion.stream_gauges import IndiaWRISClient
    return IndiaWRISClient()


def get_uptime() -> float:
    """Get service uptime in seconds."""
    return time.time() - _start_time


def check_gpu() -> tuple[bool, Optional[str]]:
    """Check GPU availability for model inference."""
    try:
        import torch
        if torch.cuda.is_available():
            return True, torch.cuda.get_device_name(0)
        return False, None
    except ImportError:
        return False, None
