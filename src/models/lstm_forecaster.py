"""
LSTM Flood Forecaster — Hindcast-Forecast Architecture (PyTorch)
=================================================================
Predicts water levels at gauge stations using a dual-LSTM with
temporal attention mechanism.

Architecture:
┌─────────────────────┐
│  Input Projection    │  Linear(n_features → d_model) + LayerNorm
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  Hindcast LSTM       │  Processes historical data → encodes system state
│  (2 layers, 128d)    │  ("What has happened up to now?")
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  Temporal Attention  │  Multi-head attention over hindcast outputs
│  (4 heads)           │  ("Which past events matter most for the forecast?")
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  Forecast LSTM       │  Takes hindcast context + future met forecasts
│  (2 layers, 128d)    │  ("Given what happened, what will happen next?")
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  Output Heads        │  Mean: Linear → T_forecast water levels
│                      │  Std:  Linear → σ (uncertainty estimation)
└─────────────────────┘

Loss: Gaussian Negative Log-Likelihood (probabilistic output)
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from config.settings import settings


# ──────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────

class FloodDataset(torch.utils.data.Dataset):
    """
    Time-series dataset with windowed input/output for LSTM training.

    Creates (input, target) pairs using a sliding window:
    - Input: [t - lookback, ..., t] dynamic features + static features
    - Target: [t + 1, ..., t + forecast_horizon] water levels

    Features per timestep:
        Dynamic: water_level, discharge, precipitation, api_3d, api_7d,
                 api_14d, api_30d, soil_moisture, temperature, ...
        Static:  catchment_area, mean_elevation, mean_slope, urban_fraction, ...
    """

    def __init__(
        self,
        dynamic_features: np.ndarray,    # (T, n_dynamic_features)
        static_features: np.ndarray,      # (n_static_features,)
        targets: np.ndarray,              # (T,) water levels
        lookback: int = 168,              # 7 days hourly
        forecast_horizon: int = 72,       # 3 days hourly
        forecast_met: Optional[np.ndarray] = None,  # (T, n_met) future met inputs
    ):
        self.dynamic = torch.FloatTensor(dynamic_features)
        self.static = torch.FloatTensor(static_features)
        self.targets = torch.FloatTensor(targets)
        self.lookback = lookback
        self.horizon = forecast_horizon
        self.forecast_met = (
            torch.FloatTensor(forecast_met) if forecast_met is not None else None
        )

        self.n_samples = len(targets) - lookback - forecast_horizon + 1
        if self.n_samples <= 0:
            raise ValueError(
                f"Not enough data: {len(targets)} timesteps for "
                f"lookback={lookback} + horizon={forecast_horizon}"
            )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Historical window
        hist_start = idx
        hist_end = idx + self.lookback

        # Forecast window
        fc_start = hist_end
        fc_end = fc_start + self.horizon

        sample = {
            "history": self.dynamic[hist_start:hist_end],          # (lookback, n_dyn)
            "static": self.static,                                  # (n_static,)
            "target": self.targets[fc_start:fc_end],               # (horizon,)
        }

        if self.forecast_met is not None:
            sample["forecast_met"] = self.forecast_met[fc_start:fc_end]  # (horizon, n_met)

        return sample


# ──────────────────────────────────────────────────────────────
# Model Components
# ──────────────────────────────────────────────────────────────

class TemporalAttention(nn.Module):
    """
    Multi-head temporal attention over LSTM hidden states.

    Allows the model to learn which historical timesteps are most
    relevant for the current forecast. During flood events, the model
    should attend to:
    - Peak rainfall periods
    - Recent rapid water level rises
    - Previous flood events with similar patterns

    Uses scaled dot-product attention with learned query projections.
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model) — LSTM hidden states

        Returns:
            (batch, seq_len, d_model) — attention-weighted states
        """
        batch, seq_len, d_model = x.shape
        residual = x

        # Multi-head projections
        Q = self.query(x).view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        K = self.key(x).view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        V = self.value(x).view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        # Scaled dot-product attention
        scale = math.sqrt(self.d_head)
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, d_model)

        # Output projection + residual + LayerNorm
        output = self.out_proj(attn_output)
        output = self.layer_norm(output + residual)

        return output


class FloodLSTM(nn.Module):
    """
    Hindcast-Forecast LSTM with Temporal Attention for flood prediction.

    Two-stage architecture:
    1. Hindcast LSTM: Processes historical observations to build
       a representation of the current hydrological state
    2. Forecast LSTM: Uses hindcast context + future meteorological
       forecasts to predict water levels

    Probabilistic output: predicts both mean and standard deviation
    of water level, enabling uncertainty quantification.
    """

    def __init__(
        self,
        n_dynamic_features: int = 12,
        n_static_features: int = 8,
        n_forecast_met_features: int = 4,
        d_model: int = 128,
        n_layers: int = 2,
        n_attention_heads: int = 4,
        forecast_horizon: int = 72,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.d_model = d_model
        self.forecast_horizon = forecast_horizon

        # Input projection: raw features → d_model
        self.input_projection = nn.Sequential(
            nn.Linear(n_dynamic_features, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Static feature embedding
        self.static_embedding = nn.Sequential(
            nn.Linear(n_static_features, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
            nn.LayerNorm(d_model),
        )

        # Hindcast LSTM — encodes historical observations
        self.hindcast_lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=False,
        )

        # Temporal attention over hindcast outputs
        self.temporal_attention = TemporalAttention(
            d_model=d_model,
            n_heads=n_attention_heads,
            dropout=dropout,
        )

        # Forecast LSTM — generates future predictions
        forecast_input_dim = d_model + n_forecast_met_features
        self.forecast_lstm = nn.LSTM(
            input_size=forecast_input_dim,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )

        # Output heads
        self.mean_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

        self.std_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Softplus(),  # Ensure positive std
        )

        # Initialize weights
        self._init_weights()

        # Log model size
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"FloodLSTM initialized | d_model={d_model} | layers={n_layers} | "
            f"params={trainable_params:,} ({total_params:,} total)"
        )

    def _init_weights(self):
        """Xavier/Kaiming initialization for stable training."""
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(
        self,
        history: torch.Tensor,           # (batch, lookback, n_dynamic)
        static: torch.Tensor,            # (batch, n_static)
        forecast_met: Optional[torch.Tensor] = None,  # (batch, horizon, n_met)
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass through the hindcast-forecast architecture.

        Args:
            history: Historical dynamic features (batch, lookback, n_features)
            static: Static catchment attributes (batch, n_static)
            forecast_met: Future meteorological forecasts (batch, horizon, n_met)

        Returns:
            Dictionary with 'mean' and 'std' predictions,
            each of shape (batch, forecast_horizon, 1)
        """
        batch_size = history.shape[0]

        # ── Input Projection ──
        x = self.input_projection(history)  # (batch, lookback, d_model)

        # ── Add static context ──
        static_emb = self.static_embedding(static)  # (batch, d_model)
        x = x + static_emb.unsqueeze(1)  # Broadcast to all timesteps

        # ── Hindcast LSTM ──
        hindcast_out, (h_n, c_n) = self.hindcast_lstm(x)
        # hindcast_out: (batch, lookback, d_model)

        # ── Temporal Attention ──
        attn_out = self.temporal_attention(hindcast_out)
        # Use the last attended state as context
        context = attn_out[:, -1:, :]  # (batch, 1, d_model)

        # ── Forecast LSTM ──
        # Repeat context for each forecast step
        context_repeated = context.repeat(1, self.forecast_horizon, 1)

        if forecast_met is not None:
            forecast_input = torch.cat([context_repeated, forecast_met], dim=-1)
        else:
            # If no met forecasts, pad with zeros
            padding = torch.zeros(
                batch_size, self.forecast_horizon, 4,  # n_forecast_met_features
                device=history.device,
            )
            forecast_input = torch.cat([context_repeated, padding], dim=-1)

        forecast_out, _ = self.forecast_lstm(
            forecast_input, (h_n, c_n)  # Initialize from hindcast state
        )
        # forecast_out: (batch, horizon, d_model)

        # ── Output Heads ──
        mean = self.mean_head(forecast_out)   # (batch, horizon, 1)
        std = self.std_head(forecast_out)     # (batch, horizon, 1)

        return {"mean": mean, "std": std}


# ──────────────────────────────────────────────────────────────
# Loss Function
# ──────────────────────────────────────────────────────────────

class GaussianNLLLoss(nn.Module):
    """
    Gaussian Negative Log-Likelihood Loss for probabilistic prediction.

    NLL = 0.5 × [log(σ²) + (y - μ)²/σ²]

    This trains the model to output both:
    - μ (mean prediction): what the most likely water level is
    - σ (standard deviation): how uncertain the prediction is

    During flood events, σ should increase — the model learns to be
    honest about its uncertainty in extreme conditions.
    """

    def __init__(self, min_std: float = 0.01):
        super().__init__()
        self.min_std = min_std

    def forward(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        std = torch.clamp(std, min=self.min_std)
        variance = std ** 2

        nll = 0.5 * (torch.log(variance) + (target - mean) ** 2 / variance)

        return nll.mean()


# ──────────────────────────────────────────────────────────────
# Inference Utilities
# ──────────────────────────────────────────────────────────────

class FloodForecaster:
    """
    High-level inference wrapper for the FloodLSTM model.

    Handles:
    - Model loading from checkpoint
    - Feature normalization (StandardScaler)
    - Batch prediction
    - Uncertainty-aware alert generation
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
    ):
        self.device = device or settings.lstm.device
        self.model = self._load_model(checkpoint_path)
        self.model.eval()

    def _load_model(self, checkpoint_path: str) -> FloodLSTM:
        """Load trained model from checkpoint."""
        logger.info(f"Loading FloodLSTM from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        model = FloodLSTM(
            n_dynamic_features=checkpoint.get("n_dynamic_features", 12),
            n_static_features=checkpoint.get("n_static_features", 8),
            d_model=checkpoint.get("d_model", 128),
            n_layers=checkpoint.get("n_layers", 2),
            n_attention_heads=checkpoint.get("n_attention_heads", 4),
            forecast_horizon=checkpoint.get("forecast_horizon", 72),
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)

        logger.info("Model loaded successfully")
        return model

    @torch.no_grad()
    def predict(
        self,
        history: np.ndarray,
        static: np.ndarray,
        forecast_met: Optional[np.ndarray] = None,
        scaler_mean: Optional[np.ndarray] = None,
        scaler_std: Optional[np.ndarray] = None,
    ) -> dict[str, np.ndarray]:
        """
        Generate flood forecast with uncertainty estimates.

        Args:
            history: (lookback, n_features) historical observations
            static: (n_static,) catchment attributes
            forecast_met: (horizon, n_met) future meteorological inputs
            scaler_mean: Feature normalization mean
            scaler_std: Feature normalization std

        Returns:
            Dictionary with 'mean', 'std', 'lower_ci', 'upper_ci' arrays
        """
        # Normalize
        if scaler_mean is not None and scaler_std is not None:
            history = (history - scaler_mean) / (scaler_std + 1e-8)

        # Convert to tensors
        h = torch.FloatTensor(history).unsqueeze(0).to(self.device)
        s = torch.FloatTensor(static).unsqueeze(0).to(self.device)
        fm = None
        if forecast_met is not None:
            fm = torch.FloatTensor(forecast_met).unsqueeze(0).to(self.device)

        # Forward pass
        output = self.model(h, s, fm)

        mean = output["mean"].squeeze().cpu().numpy()
        std = output["std"].squeeze().cpu().numpy()

        # 90% confidence interval
        z_90 = 1.645
        lower_ci = mean - z_90 * std
        upper_ci = mean + z_90 * std

        return {
            "mean": mean,
            "std": std,
            "lower_ci": lower_ci,
            "upper_ci": upper_ci,
        }

    def generate_alert(
        self,
        prediction: dict[str, np.ndarray],
        danger_level: float,
        warning_level: float,
    ) -> dict:
        """
        Generate flood alert based on prediction vs danger/warning levels.

        Alert levels (following CWC convention):
        - GREEN:  predicted peak < warning level
        - YELLOW: predicted peak approaches warning level
        - ORANGE: predicted peak exceeds warning level
        - RED:    predicted peak exceeds danger level
        """
        peak_mean = prediction["mean"].max()
        peak_upper = prediction["upper_ci"].max()
        peak_time_idx = prediction["mean"].argmax()

        if peak_upper >= danger_level:
            level = "RED"
            message = "DANGER: Water level likely to exceed danger mark"
        elif peak_mean >= danger_level:
            level = "RED"
            message = "DANGER: Water level expected to cross danger mark"
        elif peak_upper >= warning_level:
            level = "ORANGE"
            message = "WARNING: Water level may cross warning mark"
        elif peak_mean >= warning_level:
            level = "YELLOW"
            message = "CAUTION: Water level approaching warning mark"
        else:
            level = "GREEN"
            message = "NORMAL: No flood threat expected"

        return {
            "alert_level": level,
            "message": message,
            "peak_predicted_m": float(peak_mean),
            "peak_upper_ci_m": float(peak_upper),
            "peak_time_hours": int(peak_time_idx),
            "danger_level_m": danger_level,
            "warning_level_m": warning_level,
            "confidence_pct": 90,
        }
