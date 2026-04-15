"""
Centralized Metrics Logger
==========================
Handles persistence of model performance metrics into a JSON file,
which is consumed by the FastAPI backend and Streamlit dashboard.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from loguru import logger

from config.settings import MODEL_DIR

METRICS_PATH = MODEL_DIR / "training_metrics.json"

def log_metrics(model_type: str, metrics: Dict[str, Any]):
    """
    Save or update metrics for a specific model type.
    
    Args:
        model_type: 'lstm', 'xgboost', or 'system'
        metrics: Dictionary of metrics to store
    """
    data = _load_existing_metrics()
    
    # Update the specific model type
    if model_type not in data:
        data[model_type] = {}
        
    # Merge new metrics with existing ones
    data[model_type].update(metrics)
    
    # Add common fields
    data[model_type]["last_train"] = str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # Write back to file
    try:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        with open(METRICS_PATH, "w") as f:
            json.dump(data, f, indent=4)
        logger.info(f"✓ Updated {model_type} metrics in {METRICS_PATH}")
    except Exception as e:
        logger.error(f"Failed to save metrics: {e}")

def _load_existing_metrics() -> Dict[str, Any]:
    """Load existing metrics from JSON file, returning empty dict if not found."""
    if not METRICS_PATH.exists():
        return {
            "lstm": {"nse_mean": 0.0, "parameters": "7.5M", "data_source": "none"},
            "xgboost": {"auc_roc": 0.0, "feature_importance": {}, "n_features": 0, "data_source": "none"},
            "system": {"stations_monitored": 10, "is_gpu_accelerated": True}
        }
    
    try:
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to read existing metrics: {e}. Starting fresh.")
        return {}
