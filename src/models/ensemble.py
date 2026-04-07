"""
Ensemble Combiner — Temporal + Spatial Fusion
==============================================
Combines LSTM temporal predictions with XGBoost spatial susceptibility
into a unified flood risk assessment.

P_flood(x, y, t) = α × P_temporal(t) + (1 - α) × P_spatial(x, y)

- P_temporal: From LSTM — captures "when" flooding occurs
  (time-varying probability based on water levels and weather)
- P_spatial: From XGBoost — captures "where" flooding occurs
  (location-specific vulnerability based on terrain and land cover)

The fusion weight α is calibrated per watershed using held-out data.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from scipy.optimize import minimize_scalar

from config.settings import settings


class EnsembleCombiner:
    """
    Weighted fusion of temporal and spatial flood predictions.

    The key insight: temporal and spatial models capture different
    aspects of flood risk that are complementary.

    - LSTM sees: "Water levels are rising fast due to heavy rain"
    - XGBoost sees: "This location has high TWI and is near a channel"
    - Combined: "This specific location is at high risk RIGHT NOW"
    """

    def __init__(
        self,
        alpha: float = 0.6,
        calibrate_per_basin: bool = True,
    ):
        """
        Args:
            alpha: Default temporal weight (1 - alpha for spatial)
            calibrate_per_basin: Whether to learn optimal α per watershed
        """
        self.alpha = alpha
        self.calibrate_per_basin = calibrate_per_basin
        self.basin_alphas: dict[str, float] = {}

    def combine(
        self,
        p_temporal: np.ndarray | float,
        p_spatial: np.ndarray,
        basin_id: Optional[str] = None,
    ) -> np.ndarray:
        """
        Combine temporal and spatial flood probabilities.

        P_combined = α × P_temporal + (1 - α) × P_spatial

        Args:
            p_temporal: Temporal flood probability (scalar or per-timestep)
                       From LSTM — represents "how likely is flooding NOW"
            p_spatial: Spatial flood probability (per grid cell)
                      From XGBoost — represents "how vulnerable is this location"
            basin_id: Watershed identifier for basin-specific α

        Returns:
            Combined flood probability map
        """
        alpha = self.basin_alphas.get(basin_id, self.alpha) if basin_id else self.alpha

        # Broadcast temporal probability across spatial grid if scalar
        if np.isscalar(p_temporal):
            p_temporal = np.full_like(p_spatial, p_temporal)

        combined = alpha * p_temporal + (1 - alpha) * p_spatial

        # Clip to valid probability range
        combined = np.clip(combined, 0.0, 1.0)

        logger.debug(
            f"Ensemble combine | α={alpha:.2f} | "
            f"temporal_mean={np.mean(p_temporal):.4f} | "
            f"spatial_mean={np.mean(p_spatial):.4f} | "
            f"combined_mean={np.mean(combined):.4f}"
        )

        return combined

    def calibrate_alpha(
        self,
        p_temporal: np.ndarray,
        p_spatial: np.ndarray,
        y_true: np.ndarray,
        basin_id: Optional[str] = None,
    ) -> float:
        """
        Calibrate the fusion weight α using held-out validation data.

        Optimizes α to maximize Brier Skill Score:
        BSS = 1 - BS(model) / BS(climatology)

        Lower Brier Score = better calibrated probabilities.

        Args:
            p_temporal: Temporal predictions (validation set)
            p_spatial: Spatial predictions (validation set)
            y_true: True binary labels (validation set)
            basin_id: If provided, stores basin-specific α

        Returns:
            Optimal α value
        """
        from sklearn.metrics import brier_score_loss

        def neg_brier_skill(alpha):
            combined = alpha * p_temporal + (1 - alpha) * p_spatial
            combined = np.clip(combined, 0, 1)
            return brier_score_loss(y_true, combined)

        # Optimize α ∈ [0, 1]
        result = minimize_scalar(neg_brier_skill, bounds=(0, 1), method="bounded")
        optimal_alpha = result.x

        if basin_id:
            self.basin_alphas[basin_id] = optimal_alpha

        logger.info(
            f"Calibrated α={optimal_alpha:.3f} | "
            f"Brier={result.fun:.4f} | basin={basin_id}"
        )

        return optimal_alpha

    def classify_risk(
        self,
        combined_probability: np.ndarray,
        thresholds: Optional[dict[str, float]] = None,
    ) -> np.ndarray:
        """
        Classify flood risk into categorical alert levels.

        CWC-style alert scheme:
        - GREEN (0):  P < 0.3 — No significant flood risk
        - YELLOW (1): 0.3 ≤ P < 0.6 — Moderate risk, monitor
        - ORANGE (2): 0.6 ≤ P < 0.8 — High risk, prepare
        - RED (3):    P ≥ 0.8 — Severe risk, evacuate

        Args:
            combined_probability: 2D flood probability map
            thresholds: Custom threshold dict {color: min_prob}

        Returns:
            Integer risk classification map (0=green, 1=yellow, 2=orange, 3=red)
        """
        if thresholds is None:
            thresholds = settings.ensemble.alert_thresholds

        risk_map = np.zeros_like(combined_probability, dtype=np.uint8)

        # Apply thresholds (cumulative — higher overrides lower)
        risk_map[combined_probability >= thresholds["yellow"]] = 1
        risk_map[combined_probability >= thresholds["orange"]] = 2
        risk_map[combined_probability >= thresholds["red"]] = 3

        # Statistics
        total = combined_probability.size
        for level, code in [("GREEN", 0), ("YELLOW", 1), ("ORANGE", 2), ("RED", 3)]:
            count = (risk_map == code).sum()
            pct = count / total * 100
            logger.info(f"  {level}: {count:,} cells ({pct:.1f}%)")

        return risk_map

    def generate_risk_report(
        self,
        combined_probability: np.ndarray,
        risk_map: np.ndarray,
        aoi_name: str = "AOI",
    ) -> dict:
        """
        Generate summary risk report for the AOI.

        Returns structured data suitable for API response or dashboard display.
        """
        total_cells = combined_probability.size
        valid_cells = (~np.isnan(combined_probability)).sum()

        report = {
            "aoi_name": aoi_name,
            "total_cells": int(total_cells),
            "valid_cells": int(valid_cells),
            "mean_flood_probability": float(np.nanmean(combined_probability)),
            "max_flood_probability": float(np.nanmax(combined_probability)),
            "p95_flood_probability": float(np.nanpercentile(combined_probability, 95)),
            "risk_distribution": {
                "green": int((risk_map == 0).sum()),
                "yellow": int((risk_map == 1).sum()),
                "orange": int((risk_map == 2).sum()),
                "red": int((risk_map == 3).sum()),
            },
            "risk_percentages": {
                "green": float((risk_map == 0).mean() * 100),
                "yellow": float((risk_map == 1).mean() * 100),
                "orange": float((risk_map == 2).mean() * 100),
                "red": float((risk_map == 3).mean() * 100),
            },
        }

        return report
