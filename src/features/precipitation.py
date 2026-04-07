"""
Antecedent Precipitation Index (API) Computation
=================================================
The API quantifies soil wetness based on recent rainfall history.

API_t = k × API_{t-1} + P_t

where:
    k = decay factor (0.85-0.98, soil-dependent)
    P_t = daily rainfall (mm) on day t

Physical meaning:
- High API = soil is already wet → low infiltration capacity
- New rainfall runs off quickly → flood risk amplified
- k encodes how fast the soil dries out (sandy soil → lower k, clay → higher k)

Multi-scale APIs (3-day, 7-day, 14-day, 30-day) capture both
flash flood triggers (short-term) and sustained flooding (long-term).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger


class AntecedentPrecipitationIndex:
    """
    Compute Antecedent Precipitation Index at multiple time scales.

    The API is a simple but powerful physical proxy for soil saturation.
    It's the "memory" of the landscape — how much of the recent rain
    is still sitting in the ground.
    """

    # Decay factors by soil type (approximate values)
    SOIL_DECAY_FACTORS = {
        "sandy": 0.85,
        "sandy_loam": 0.88,
        "loam": 0.90,
        "clay_loam": 0.93,
        "clay": 0.95,
        "organic": 0.92,
        "default": 0.90,
    }

    def __init__(self, decay_factor: float = 0.90):
        """
        Args:
            decay_factor: k in API formula. Range [0.85, 0.98].
                Higher k = slower drying = heavier soils.
        """
        self.k = decay_factor

    def compute_api_timeseries(
        self,
        precipitation: pd.Series | np.ndarray,
        k: Optional[float] = None,
        initial_api: float = 0.0,
    ) -> np.ndarray:
        """
        Compute daily API time series from precipitation data.

        API_t = k × API_{t-1} + P_t

        This is computed iteratively (not vectorizable) because each
        day depends on the previous day's state.

        Args:
            precipitation: Daily precipitation (mm) — 1D array or Series
            k: Override decay factor (uses instance default if None)
            initial_api: Starting API value (mm)

        Returns:
            API time series (same length as precipitation)
        """
        if k is None:
            k = self.k

        precip = np.asarray(precipitation, dtype=np.float64)
        n = len(precip)

        api = np.zeros(n, dtype=np.float64)
        api[0] = k * initial_api + np.nan_to_num(precip[0])

        for t in range(1, n):
            api[t] = k * api[t - 1] + np.nan_to_num(precip[t])

        return api

    def compute_multi_scale_api(
        self,
        precipitation: pd.Series | np.ndarray,
        windows: list[int] | None = None,
    ) -> pd.DataFrame:
        """
        Compute API at multiple time scales.

        Each scale uses a different effective decay factor:
        - Short windows (3-day): effectively k^3 ≈ recent memory
        - Long windows (30-day): k^30 ≈ deep soil moisture proxy

        Args:
            precipitation: Daily precipitation (mm)
            windows: List of window sizes in days (default: [3, 7, 14, 30])

        Returns:
            DataFrame with columns: api_3d, api_7d, api_14d, api_30d
        """
        if windows is None:
            windows = [3, 7, 14, 30]

        precip = np.asarray(precipitation, dtype=np.float64)

        result = {}
        for window in windows:
            # Weighted sum: recent days count more
            weights = np.array([self.k ** i for i in range(window)])
            weights = weights[::-1]  # Most recent has highest weight

            # Convolution-based API (equivalent to recursive for stationary k)
            api_values = np.convolve(np.nan_to_num(precip), weights, mode="full")[:len(precip)]

            # Handle edge effects: first 'window' days are partial
            result[f"api_{window}d"] = api_values

        df = pd.DataFrame(result)
        logger.info(
            f"Multi-scale API computed | windows={windows} | "
            f"max_api_30d={df['api_30d'].max():.1f} mm"
        )

        return df

    def compute_spatial_api(
        self,
        rainfall_grid: xr.DataArray,
        window_days: int = 14,
    ) -> xr.DataArray:
        """
        Compute spatially distributed API from gridded rainfall data.

        Operates on IMD/GPM gridded rainfall (lat, lon, time) to produce
        an API grid at each timestep.

        Args:
            rainfall_grid: xarray DataArray with dims (time, latitude, longitude)
                           Units: mm/day
            window_days: API window in days

        Returns:
            xarray DataArray with same spatial dims, API values in mm
        """
        logger.info(f"Computing spatial API | window={window_days}d | shape={rainfall_grid.shape}")

        # Compute rolling weighted sum
        weights = xr.DataArray(
            [self.k ** i for i in range(window_days)][::-1],
            dims=["window"],
        )

        # Rolling window along time dimension
        rolling = rainfall_grid.rolling(time=window_days, min_periods=1)
        api_grid = rolling.construct("window").dot(weights[:window_days])

        api_grid.attrs["units"] = "mm"
        api_grid.attrs["long_name"] = f"Antecedent Precipitation Index ({window_days}-day)"
        api_grid.attrs["decay_factor"] = self.k

        logger.info(
            f"Spatial API computed | mean={float(api_grid.mean()):.1f} mm | "
            f"max={float(api_grid.max()):.1f} mm"
        )

        return api_grid

    @staticmethod
    def classify_soil_moisture_condition(api_value: float) -> str:
        """
        Classify antecedent moisture condition (AMC) from API.

        Based on SCS-CN method (US Soil Conservation Service):
        - AMC I (Dry): API < 13 mm → lowest runoff potential
        - AMC II (Normal): 13 ≤ API ≤ 28 mm → average conditions
        - AMC III (Wet): API > 28 mm → highest runoff potential

        These thresholds are approximate and should be calibrated
        for Indian conditions (monsoon baseline is much higher).
        """
        if api_value < 13:
            return "AMC_I_DRY"
        elif api_value <= 28:
            return "AMC_II_NORMAL"
        else:
            return "AMC_III_WET"
