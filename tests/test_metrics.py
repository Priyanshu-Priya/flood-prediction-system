"""Tests for hydrological evaluation metrics."""

import numpy as np
import pytest

from src.evaluation.metrics import HydrologicalMetrics, SpatialMetrics


class TestNSE:
    """Test Nash-Sutcliffe Efficiency."""

    def test_perfect_prediction(self):
        obs = np.array([1, 2, 3, 4, 5], dtype=float)
        sim = np.array([1, 2, 3, 4, 5], dtype=float)
        assert HydrologicalMetrics.nash_sutcliffe(obs, sim) == pytest.approx(1.0)

    def test_mean_prediction(self):
        """Predicting the mean of observations gives NSE = 0."""
        obs = np.array([1, 2, 3, 4, 5], dtype=float)
        sim = np.full(5, np.mean(obs))
        assert HydrologicalMetrics.nash_sutcliffe(obs, sim) == pytest.approx(0.0)

    def test_bad_prediction_negative_nse(self):
        """A bad model should have NSE < 0."""
        obs = np.array([1, 2, 3, 4, 5], dtype=float)
        sim = np.array([10, 20, 30, 40, 50], dtype=float)
        nse = HydrologicalMetrics.nash_sutcliffe(obs, sim)
        assert nse < 0

    def test_handles_nan(self):
        obs = np.array([1, np.nan, 3, 4, 5], dtype=float)
        sim = np.array([1, 2, 3, np.nan, 5], dtype=float)
        nse = HydrologicalMetrics.nash_sutcliffe(obs, sim)
        assert np.isfinite(nse)


class TestKGE:
    """Test Kling-Gupta Efficiency."""

    def test_perfect_kge(self):
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = HydrologicalMetrics.kling_gupta(obs, obs)
        assert result["kge"] == pytest.approx(1.0)
        assert result["r"] == pytest.approx(1.0)
        assert result["alpha"] == pytest.approx(1.0)
        assert result["beta"] == pytest.approx(1.0)

    def test_biased_prediction(self):
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sim = obs * 2  # 2× overestimation
        result = HydrologicalMetrics.kling_gupta(obs, sim)
        assert result["beta"] == pytest.approx(2.0, rel=0.01)
        assert result["kge"] < 1.0


class TestRMSE:
    def test_perfect(self):
        obs = np.array([1, 2, 3])
        assert HydrologicalMetrics.rmse(obs, obs) == pytest.approx(0.0)

    def test_known_value(self):
        obs = np.array([1.0, 2.0, 3.0])
        sim = np.array([2.0, 3.0, 4.0])
        assert HydrologicalMetrics.rmse(obs, sim) == pytest.approx(1.0)


class TestPercentBias:
    def test_no_bias(self):
        obs = np.array([10, 20, 30])
        assert HydrologicalMetrics.percent_bias(obs, obs) == pytest.approx(0.0)

    def test_positive_bias(self):
        obs = np.array([10.0, 20.0, 30.0])
        sim = np.array([12.0, 24.0, 36.0])
        pbias = HydrologicalMetrics.percent_bias(obs, sim)
        assert pbias > 0  # Overestimation


class TestSpatialMetrics:
    def test_iou_perfect(self):
        mask = np.array([[1, 0], [0, 1]], dtype=bool)
        assert SpatialMetrics.flood_extent_iou(mask, mask) == pytest.approx(1.0)

    def test_iou_no_overlap(self):
        pred = np.array([[1, 0], [0, 0]], dtype=bool)
        obs = np.array([[0, 0], [0, 1]], dtype=bool)
        assert SpatialMetrics.flood_extent_iou(pred, obs) == pytest.approx(0.0)

    def test_pod_all_detected(self):
        pred = np.array([[1, 1], [1, 1]])
        obs = np.array([[1, 0], [1, 0]])
        assert SpatialMetrics.probability_of_detection(pred, obs) == pytest.approx(1.0)

    def test_far_no_false_alarms(self):
        pred = np.array([[1, 0], [0, 0]])
        obs = np.array([[1, 0], [0, 0]])
        assert SpatialMetrics.false_alarm_ratio(pred, obs) == pytest.approx(0.0)
