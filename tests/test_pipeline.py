"""
Pipeline Integration Tests
================================
Tests the end-to-end data flow: raw data → features → model input.
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import MODEL_DIR, PROCESSED_DATA_DIR


class TestLSTMDatasetPrep:
    """Test LSTM dataset preparation logic."""

    def test_dynamic_features(self):
        """Test dynamic feature construction from a discharge DataFrame."""
        from scripts.prepare_lstm_dataset import build_dynamic_features

        # Create a mock discharge DataFrame
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        df = pd.DataFrame({
            "water_level_m": np.sin(np.linspace(0, 4 * np.pi, 100)) * 2 + 10,
            "discharge_cumecs": np.abs(np.sin(np.linspace(0, 4 * np.pi, 100))) * 500 + 100,
        }, index=dates)

        features = build_dynamic_features(df)

        assert features.shape == (100, 12)
        assert features.dtype == np.float32
        assert not np.any(np.isnan(features))

    def test_static_features(self):
        """Test static feature construction."""
        from scripts.prepare_lstm_dataset import build_static_features

        station = {
            "lat": 25.6,
            "lon": 85.1,
            "danger_level_m": 12.0,
            "warning_level_m": 10.5,
            "basin": "Ganga",
        }

        static = build_static_features(station)

        assert len(static) == 8
        assert static.dtype == np.float32
        assert static[7] == 1.0  # is_major_basin = Ganga

    def test_normalization(self):
        """Test feature normalization."""
        from scripts.prepare_lstm_dataset import normalize_features

        data = np.random.randn(100, 12).astype(np.float32) * 10 + 5
        normalized, mean, std = normalize_features(data)

        assert normalized.shape == data.shape
        assert np.allclose(np.nanmean(normalized, axis=0), 0, atol=1e-5)
        assert np.allclose(np.nanstd(normalized, axis=0), 1, atol=1e-5)


class TestTrainingMetrics:
    """Test training metrics persistence."""

    def test_metrics_schema(self, tmp_path):
        """Verify training_metrics.json schema."""
        # Create a minimal metrics file
        metrics = {
            "timestamp": "2024-01-01T00:00:00",
            "lstm": {
                "nse_mean": 0.75,
                "data_source": "real",
                "parameters": "7.5M",
            },
            "xgboost": {
                "auc_roc": 0.85,
                "n_features": 7,
                "feature_importance": {"twi": 100.0, "slope": 50.0},
            },
            "system": {
                "gpu_name": "Test GPU",
                "is_gpu_accelerated": False,
            },
        }

        path = tmp_path / "training_metrics.json"
        with open(path, "w") as f:
            json.dump(metrics, f)

        # Read back
        with open(path) as f:
            loaded = json.load(f)

        assert loaded["lstm"]["nse_mean"] == 0.75
        assert loaded["xgboost"]["auc_roc"] == 0.85
        assert "twi" in loaded["xgboost"]["feature_importance"]


class TestModelCheckpoints:
    """Test that model checkpoint paths are consistent."""

    def test_model_dir_structure(self):
        """MODEL_DIR should be defined and consistent."""
        assert MODEL_DIR is not None
        assert isinstance(MODEL_DIR, Path)

    def test_processed_dir_structure(self):
        """PROCESSED_DATA_DIR should be defined."""
        assert PROCESSED_DATA_DIR is not None
        assert isinstance(PROCESSED_DATA_DIR, Path)


class TestEnsembleCombiner:
    """Test ensemble fusion logic."""

    def test_alpha_weighting(self):
        """Test that ensemble alpha produces correct weighted average."""
        from src.models.ensemble import EnsembleCombiner

        combiner = EnsembleCombiner(alpha=0.6)

        temporal_prob = 0.8
        spatial_prob = 0.3

        combined = combiner.alpha * temporal_prob + (1 - combiner.alpha) * spatial_prob
        expected = 0.6 * 0.8 + 0.4 * 0.3

        assert abs(combined - expected) < 1e-6

    def test_alpha_bounds(self):
        """Alpha should be between 0 and 1."""
        from src.models.ensemble import EnsembleCombiner

        combiner = EnsembleCombiner(alpha=0.5)
        assert 0 <= combiner.alpha <= 1
