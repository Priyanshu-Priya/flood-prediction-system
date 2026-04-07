"""
Hydrological & ML Evaluation Metrics
======================================
Comprehensive metrics for validating both temporal (LSTM) and
spatial (XGBoost) flood predictions.

Temporal metrics (water level forecasting):
- NSE (Nash-Sutcliffe Efficiency): Standard hydrological goodness-of-fit
- KGE (Kling-Gupta Efficiency): Better decomposition than NSE
- RMSE, MAE, PBIAS: Standard regression metrics

Spatial metrics (flood susceptibility mapping):
- AUC-ROC: Discrimination ability
- Brier Score: Calibration quality
- IoU (Intersection over Union): Flood extent accuracy

NSE = 1 - Σ(Qo - Qs)² / Σ(Qo - Q̄o)²
- NSE = 1.0: Perfect prediction
- NSE > 0.75: Very good
- NSE = 0.0: Model is no better than using the mean
- NSE < 0: Model is worse than the mean (garbage)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


class HydrologicalMetrics:
    """
    Hydrological evaluation metrics for streamflow / water level prediction.

    These are the standard metrics used by the hydrology community
    to evaluate model predictions. A paper without NSE/KGE won't
    get published in any serious journal.
    """

    @staticmethod
    def nash_sutcliffe(
        observed: np.ndarray,
        simulated: np.ndarray,
    ) -> float:
        """
        Nash-Sutcliffe Efficiency (NSE).

        NSE = 1 - Σ(Qo - Qs)² / Σ(Qo - Q̄o)²

        Ranges: (-∞, 1.0]
            1.0  = perfect
            0.0  = model = mean of observations
            <0   = model is worse than the mean

        This is THE standard metric in hydrology. If your NSE > 0.75,
        you have a publishable model for the Indian monsoon.
        """
        observed = np.asarray(observed, dtype=np.float64).flatten()
        simulated = np.asarray(simulated, dtype=np.float64).flatten()

        # Remove NaN pairs
        mask = ~(np.isnan(observed) | np.isnan(simulated))
        obs = observed[mask]
        sim = simulated[mask]

        if len(obs) < 2:
            return np.nan

        numerator = np.sum((obs - sim) ** 2)
        denominator = np.sum((obs - np.mean(obs)) ** 2)

        if denominator == 0:
            return np.nan

        nse = 1.0 - (numerator / denominator)
        return float(nse)

    @staticmethod
    def kling_gupta(
        observed: np.ndarray,
        simulated: np.ndarray,
    ) -> dict[str, float]:
        """
        Kling-Gupta Efficiency (KGE) and its components.

        KGE = 1 - √((r - 1)² + (α - 1)² + (β - 1)²)

        where:
            r = Pearson correlation (timing)
            α = σ_sim / σ_obs (variability ratio)
            β = μ_sim / μ_obs (bias ratio)

        KGE decomposes model performance into three independent
        components: correlation, variability, and bias.
        This is more informative than NSE alone.

        Ranges: (-∞, 1.0]
            > 0.0 generally indicates better than mean
            > 0.70 is considered good
        """
        obs = np.asarray(observed, dtype=np.float64).flatten()
        sim = np.asarray(simulated, dtype=np.float64).flatten()

        mask = ~(np.isnan(obs) | np.isnan(sim))
        obs, sim = obs[mask], sim[mask]

        if len(obs) < 2:
            return {"kge": np.nan, "r": np.nan, "alpha": np.nan, "beta": np.nan}

        # Correlation
        r = np.corrcoef(obs, sim)[0, 1]

        # Variability ratio
        alpha = np.std(sim) / (np.std(obs) + 1e-10)

        # Bias ratio
        beta = np.mean(sim) / (np.mean(obs) + 1e-10)

        # KGE
        kge = 1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

        return {
            "kge": float(kge),
            "r": float(r),          # Correlation component
            "alpha": float(alpha),   # Variability component
            "beta": float(beta),     # Bias component
        }

    @staticmethod
    def rmse(observed: np.ndarray, simulated: np.ndarray) -> float:
        """Root Mean Square Error."""
        obs = np.asarray(observed).flatten()
        sim = np.asarray(simulated).flatten()
        mask = ~(np.isnan(obs) | np.isnan(sim))
        return float(np.sqrt(np.mean((obs[mask] - sim[mask]) ** 2)))

    @staticmethod
    def mae(observed: np.ndarray, simulated: np.ndarray) -> float:
        """Mean Absolute Error."""
        obs = np.asarray(observed).flatten()
        sim = np.asarray(simulated).flatten()
        mask = ~(np.isnan(obs) | np.isnan(sim))
        return float(np.mean(np.abs(obs[mask] - sim[mask])))

    @staticmethod
    def percent_bias(observed: np.ndarray, simulated: np.ndarray) -> float:
        """
        Percent Bias (PBIAS).

        PBIAS = 100 × Σ(Qs - Qo) / Σ(Qo)

        Positive PBIAS: model overestimates (wet bias)
        Negative PBIAS: model underestimates (dry bias)
        |PBIAS| < 25% is generally acceptable
        """
        obs = np.asarray(observed).flatten()
        sim = np.asarray(simulated).flatten()
        mask = ~(np.isnan(obs) | np.isnan(sim))
        obs, sim = obs[mask], sim[mask]

        if np.sum(obs) == 0:
            return np.nan

        return float(100 * np.sum(sim - obs) / np.sum(obs))

    @staticmethod
    def peak_error(observed: np.ndarray, simulated: np.ndarray) -> dict[str, float]:
        """
        Peak flow error analysis.

        For flood forecasting, getting the peak right is critical.
        This computes errors in both peak magnitude and timing.
        """
        obs = np.asarray(observed).flatten()
        sim = np.asarray(simulated).flatten()

        peak_obs = np.max(obs)
        peak_sim = np.max(sim)
        peak_obs_idx = np.argmax(obs)
        peak_sim_idx = np.argmax(sim)

        return {
            "peak_magnitude_error_pct": float(
                (peak_sim - peak_obs) / (peak_obs + 1e-10) * 100
            ),
            "peak_timing_error_steps": int(peak_sim_idx - peak_obs_idx),
            "peak_observed": float(peak_obs),
            "peak_simulated": float(peak_sim),
        }

    @staticmethod
    def lead_time_degradation(
        observed: np.ndarray,
        predictions_by_lead: dict[int, np.ndarray],
    ) -> pd.DataFrame:
        """
        Analyze how model performance degrades with forecast lead time.

        Plots NSE vs lead time to determine the useful forecast horizon.
        A flood model that's good at 6h but garbage at 48h isn't useful
        for evacuation planning (which needs 24-48h notice).

        Args:
            observed: Ground truth water levels
            predictions_by_lead: {lead_time_hours: prediction_array}

        Returns:
            DataFrame with lead_time, nse, rmse, kge columns
        """
        results = []

        for lead_hours, pred in sorted(predictions_by_lead.items()):
            nse = HydrologicalMetrics.nash_sutcliffe(observed, pred)
            rmse_val = HydrologicalMetrics.rmse(observed, pred)
            kge = HydrologicalMetrics.kling_gupta(observed, pred)

            results.append({
                "lead_time_hours": lead_hours,
                "nse": nse,
                "rmse": rmse_val,
                "kge": kge["kge"],
            })

        df = pd.DataFrame(results)

        logger.info(
            f"Lead-time degradation analysis:\n"
            f"{df.to_string(index=False)}"
        )

        return df


class SpatialMetrics:
    """
    Evaluation metrics for spatial flood extent / susceptibility mapping.
    """

    @staticmethod
    def flood_extent_iou(
        predicted_mask: np.ndarray,
        observed_mask: np.ndarray,
    ) -> float:
        """
        Intersection over Union for flood extent comparison.

        IoU = |Predicted ∩ Observed| / |Predicted ∪ Observed|

        Compares SAR-derived flood extents (ground truth) against
        model-predicted flood zones. IoU > 0.5 is decent, > 0.7 is good.
        """
        pred = predicted_mask.astype(bool)
        obs = observed_mask.astype(bool)

        intersection = np.logical_and(pred, obs).sum()
        union = np.logical_or(pred, obs).sum()

        if union == 0:
            return 1.0 if intersection == 0 else 0.0

        iou = intersection / union
        return float(iou)

    @staticmethod
    def false_alarm_ratio(
        predicted_mask: np.ndarray,
        observed_mask: np.ndarray,
    ) -> float:
        """
        False Alarm Ratio (FAR).

        FAR = FP / (FP + TP)

        FAR < 0.25 is the target for actionable flood alerts.
        High FAR means the public loses trust in warnings.
        """
        pred = predicted_mask.astype(bool)
        obs = observed_mask.astype(bool)

        tp = np.logical_and(pred, obs).sum()
        fp = np.logical_and(pred, ~obs).sum()

        if (fp + tp) == 0:
            return 0.0

        return float(fp / (fp + tp))

    @staticmethod
    def probability_of_detection(
        predicted_mask: np.ndarray,
        observed_mask: np.ndarray,
    ) -> float:
        """
        Probability of Detection (POD / Hit Rate).

        POD = TP / (TP + FN)

        POD > 0.85 means we catch 85% of actual floods.
        Missing a real flood (FN) can cost lives.
        """
        pred = predicted_mask.astype(bool)
        obs = observed_mask.astype(bool)

        tp = np.logical_and(pred, obs).sum()
        fn = np.logical_and(~pred, obs).sum()

        if (tp + fn) == 0:
            return 1.0

        return float(tp / (tp + fn))

    @staticmethod
    def critical_success_index(
        predicted_mask: np.ndarray,
        observed_mask: np.ndarray,
    ) -> float:
        """
        Critical Success Index (CSI / Threat Score).

        CSI = TP / (TP + FP + FN)

        Combines both false alarms and misses into one score.
        CSI > 0.3 is typical for operational flood forecasts.
        """
        pred = predicted_mask.astype(bool)
        obs = observed_mask.astype(bool)

        tp = np.logical_and(pred, obs).sum()
        fp = np.logical_and(pred, ~obs).sum()
        fn = np.logical_and(~pred, obs).sum()

        denom = tp + fp + fn
        if denom == 0:
            return 1.0

        return float(tp / denom)

    @staticmethod
    def full_contingency_report(
        predicted_mask: np.ndarray,
        observed_mask: np.ndarray,
    ) -> dict:
        """Full contingency table analysis for flood extent validation."""
        pred = predicted_mask.astype(bool)
        obs = observed_mask.astype(bool)

        tp = np.logical_and(pred, obs).sum()
        tn = np.logical_and(~pred, ~obs).sum()
        fp = np.logical_and(pred, ~obs).sum()
        fn = np.logical_and(~pred, obs).sum()

        total = tp + tn + fp + fn

        return {
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "accuracy": float((tp + tn) / total) if total > 0 else 0,
            "iou": SpatialMetrics.flood_extent_iou(predicted_mask, observed_mask),
            "far": SpatialMetrics.false_alarm_ratio(predicted_mask, observed_mask),
            "pod": SpatialMetrics.probability_of_detection(predicted_mask, observed_mask),
            "csi": SpatialMetrics.critical_success_index(predicted_mask, observed_mask),
        }
