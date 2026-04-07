"""Tests for LSTM model architecture and forward pass."""

import numpy as np
import pytest
import torch


class TestFloodLSTM:
    """Test the FloodLSTM model."""

    def _create_model(self, d_model=64, n_layers=1):
        from src.models.lstm_forecaster import FloodLSTM
        return FloodLSTM(
            n_dynamic_features=8,
            n_static_features=4,
            n_forecast_met_features=3,
            d_model=d_model,
            n_layers=n_layers,
            n_attention_heads=4,
            forecast_horizon=24,
            dropout=0.1,
        )

    def test_forward_pass_shape(self):
        """Output shape should match (batch, horizon, 1)."""
        model = self._create_model()
        batch_size, lookback = 4, 48

        history = torch.randn(batch_size, lookback, 8)
        static = torch.randn(batch_size, 4)
        forecast_met = torch.randn(batch_size, 24, 3)

        output = model(history, static, forecast_met)

        assert output["mean"].shape == (batch_size, 24, 1)
        assert output["std"].shape == (batch_size, 24, 1)

    def test_std_is_positive(self):
        """Softplus output head should ensure positive std."""
        model = self._create_model()
        history = torch.randn(2, 48, 8)
        static = torch.randn(2, 4)

        output = model(history, static)

        assert torch.all(output["std"] > 0), "Standard deviation must be positive"

    def test_no_forecast_met(self):
        """Model should work without forecast meteorological inputs."""
        model = self._create_model()
        history = torch.randn(2, 48, 8)
        static = torch.randn(2, 4)

        output = model(history, static, forecast_met=None)
        assert output["mean"].shape == (2, 24, 1)

    def test_gradient_flow(self):
        """Gradients should flow through all model components."""
        model = self._create_model()
        history = torch.randn(2, 48, 8, requires_grad=True)
        static = torch.randn(2, 4)

        output = model(history, static)
        loss = output["mean"].sum()
        loss.backward()

        assert history.grad is not None
        assert not torch.all(history.grad == 0)


class TestGaussianNLLLoss:
    """Test the probabilistic loss function."""

    def test_perfect_prediction_low_loss(self):
        """When mean matches target, loss should be small."""
        from src.models.lstm_forecaster import GaussianNLLLoss

        criterion = GaussianNLLLoss()
        target = torch.tensor([1.0, 2.0, 3.0])
        mean = torch.tensor([1.0, 2.0, 3.0])
        std = torch.tensor([0.1, 0.1, 0.1])

        loss = criterion(mean, std, target)
        assert loss.item() < 1.0

    def test_large_error_high_loss(self):
        """Large prediction errors should produce high loss."""
        from src.models.lstm_forecaster import GaussianNLLLoss

        criterion = GaussianNLLLoss()
        target = torch.tensor([1.0])
        mean = torch.tensor([100.0])
        std = torch.tensor([0.1])

        loss = criterion(mean, std, target)
        assert loss.item() > 10


class TestFloodDataset:
    """Test the time-series dataset."""

    def test_dataset_length(self):
        from src.models.lstm_forecaster import FloodDataset

        n_timesteps = 500
        dynamic = np.random.randn(n_timesteps, 8).astype(np.float32)
        static = np.random.randn(4).astype(np.float32)
        targets = np.random.randn(n_timesteps).astype(np.float32)

        dataset = FloodDataset(dynamic, static, targets, lookback=48, forecast_horizon=24)

        expected_len = n_timesteps - 48 - 24 + 1
        assert len(dataset) == expected_len

    def test_sample_shapes(self):
        from src.models.lstm_forecaster import FloodDataset

        dynamic = np.random.randn(200, 8).astype(np.float32)
        static = np.random.randn(4).astype(np.float32)
        targets = np.random.randn(200).astype(np.float32)

        dataset = FloodDataset(dynamic, static, targets, lookback=48, forecast_horizon=24)
        sample = dataset[0]

        assert sample["history"].shape == (48, 8)
        assert sample["static"].shape == (4,)
        assert sample["target"].shape == (24,)
