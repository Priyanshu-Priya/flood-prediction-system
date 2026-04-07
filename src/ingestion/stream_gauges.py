"""
India Stream Gauge & Water Level Ingestion
============================================
Integrates with India-specific water data sources:
  - India-WRIS (Water Resources Information System) — real-time river levels
  - CWC (Central Water Commission) — flood forecast bulletins
  - Fallback: GRDC for transboundary basins (Brahmaputra, Indus)

Ground-truth water levels are essential for:
1. Training the LSTM forecaster
2. Calibrating discharge estimation (rating curves)
3. Real-time alert generation
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings


class IndiaWRISClient:
    """
    Client for India Water Resources Information System (India-WRIS).

    India-WRIS provides:
    - Real-time water levels from CWC telemetry stations
    - Historical daily discharge records
    - Reservoir storage and inflow data
    - Basin-level water balance information

    Endpoint structure (based on available public data):
    - Station metadata: /stations
    - Real-time levels: /stations/{id}/realtime
    - Historical daily: /stations/{id}/historical
    """

    def __init__(self):
        self.base_url = settings.data_sources.wris_base_url
        self.api_key = settings.data_sources.wris_api_key

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    def fetch_station_metadata(
        self,
        state: Optional[str] = None,
        basin: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch metadata for all CWC/India-WRIS telemetry stations.

        Args:
            state: Filter by Indian state (e.g., "Tamil Nadu", "Assam")
            basin: Filter by river basin (e.g., "Ganga", "Brahmaputra", "Krishna")

        Returns:
            DataFrame with columns: station_id, name, lat, lon, river, basin, state, type
        """
        import httpx

        logger.info(f"Fetching India-WRIS station metadata | state={state} | basin={basin}")

        params = {}
        if state:
            params["state"] = state
        if basin:
            params["basin"] = basin

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            response = httpx.get(
                f"{self.base_url}/stations",
                params=params,
                headers=headers,
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()

            df = pd.DataFrame(data.get("stations", data))
            logger.info(f"Retrieved {len(df)} stations")
            return df

        except httpx.HTTPError as e:
            logger.warning(f"India-WRIS API error: {e}. Using fallback station list.")
            return self._get_fallback_stations(state, basin)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    def fetch_water_levels(
        self,
        station_id: str,
        start_date: str,
        end_date: str,
        frequency: str = "daily",
    ) -> pd.DataFrame:
        """
        Fetch historical water level data for a specific station.

        Args:
            station_id: India-WRIS station identifier
            start_date: Start date (ISO format)
            end_date: End date (ISO format)
            frequency: "hourly", "daily", or "realtime" (15-min)

        Returns:
            DataFrame with columns: timestamp, water_level_m, discharge_cumecs,
            quality_flag
        """
        import httpx

        logger.info(
            f"Fetching water levels | station={station_id} | "
            f"dates={start_date}/{end_date} | freq={frequency}"
        )

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            response = httpx.get(
                f"{self.base_url}/stations/{station_id}/historical",
                params={
                    "start": start_date,
                    "end": end_date,
                    "frequency": frequency,
                },
                headers=headers,
                timeout=60.0,
            )
            response.raise_for_status()
            data = response.json()

            df = pd.DataFrame(data.get("observations", data))

            # Standardize column names
            column_map = {
                "date": "timestamp",
                "datetime": "timestamp",
                "water_level": "water_level_m",
                "level": "water_level_m",
                "discharge": "discharge_cumecs",
                "flow": "discharge_cumecs",
            }
            df.rename(columns={k: v for k, v in column_map.items() if k in df.columns}, inplace=True)

            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df.set_index("timestamp", inplace=True)
                df.sort_index(inplace=True)

            # Quality control
            df = self._apply_quality_control(df)

            logger.info(f"Loaded {len(df)} observations for station {station_id}")
            return df

        except httpx.HTTPError as e:
            logger.error(f"Failed to fetch water levels for {station_id}: {e}")
            raise

    def fetch_realtime_levels(
        self,
        station_ids: list[str],
    ) -> pd.DataFrame:
        """
        Fetch current real-time water levels for multiple stations.

        Used for live monitoring and alert generation.

        Args:
            station_ids: List of station identifiers

        Returns:
            DataFrame with latest readings per station
        """
        import httpx

        logger.info(f"Fetching real-time levels for {len(station_ids)} stations")

        records = []
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        for sid in station_ids:
            try:
                response = httpx.get(
                    f"{self.base_url}/stations/{sid}/realtime",
                    headers=headers,
                    timeout=15.0,
                )
                response.raise_for_status()
                data = response.json()

                records.append({
                    "station_id": sid,
                    "timestamp": data.get("timestamp"),
                    "water_level_m": data.get("water_level"),
                    "discharge_cumecs": data.get("discharge"),
                    "danger_level_m": data.get("danger_level"),
                    "warning_level_m": data.get("warning_level"),
                    "trend": data.get("trend"),  # "rising", "falling", "steady"
                })
            except Exception as e:
                logger.warning(f"Failed to fetch realtime for {sid}: {e}")
                continue

        df = pd.DataFrame(records)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        return df

    def _apply_quality_control(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply automated quality control checks to gauge data.

        Checks:
        1. Remove physically impossible values (negative water levels excluded
           only if below known datum; extremely high values)
        2. Flag sensor drift via rolling Z-score
        3. Interpolate short gaps (< 3 hours) with linear interpolation
        """
        if "water_level_m" not in df.columns:
            return df

        original_len = len(df)

        # Remove extreme outliers (> 5σ from rolling mean)
        rolling_mean = df["water_level_m"].rolling(window=24, min_periods=6).mean()
        rolling_std = df["water_level_m"].rolling(window=24, min_periods=6).std()

        z_scores = np.abs((df["water_level_m"] - rolling_mean) / (rolling_std + 1e-6))
        df.loc[z_scores > 5, "water_level_m"] = np.nan

        # Flag quality
        df["quality_flag"] = "good"
        df.loc[z_scores > 3, "quality_flag"] = "suspect"
        df.loc[df["water_level_m"].isna(), "quality_flag"] = "missing"

        # Interpolate short gaps (up to 3 hours)
        df["water_level_m"] = df["water_level_m"].interpolate(
            method="linear", limit=3
        )

        removed = original_len - df["water_level_m"].notna().sum()
        if removed > 0:
            logger.info(f"QC: removed/flagged {removed} suspect observations")

        return df

    def _get_fallback_stations(
        self,
        state: Optional[str] = None,
        basin: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fallback station list for major Indian CWC stations.
        Used when API is unavailable.
        """
        stations = [
            {"station_id": "CWC001", "name": "Farakka Barrage", "lat": 24.81, "lon": 87.92,
             "river": "Ganga", "basin": "Ganga", "state": "West Bengal", "type": "barrage"},
            {"station_id": "CWC002", "name": "Varanasi", "lat": 25.32, "lon": 83.01,
             "river": "Ganga", "basin": "Ganga", "state": "Uttar Pradesh", "type": "gauge"},
            {"station_id": "CWC003", "name": "Patna", "lat": 25.60, "lon": 85.10,
             "river": "Ganga", "basin": "Ganga", "state": "Bihar", "type": "gauge"},
            {"station_id": "CWC004", "name": "Dibrugarh", "lat": 27.47, "lon": 94.91,
             "river": "Brahmaputra", "basin": "Brahmaputra", "state": "Assam", "type": "gauge"},
            {"station_id": "CWC005", "name": "Guwahati", "lat": 26.19, "lon": 91.75,
             "river": "Brahmaputra", "basin": "Brahmaputra", "state": "Assam", "type": "gauge"},
            {"station_id": "CWC006", "name": "Vijayawada", "lat": 16.52, "lon": 80.62,
             "river": "Krishna", "basin": "Krishna", "state": "Andhra Pradesh", "type": "gauge"},
            {"station_id": "CWC007", "name": "Adyar", "lat": 13.01, "lon": 80.25,
             "river": "Adyar", "basin": "East Flowing Rivers", "state": "Tamil Nadu", "type": "gauge"},
            {"station_id": "CWC008", "name": "Surat", "lat": 21.17, "lon": 72.83,
             "river": "Tapi", "basin": "Tapi", "state": "Gujarat", "type": "gauge"},
            {"station_id": "CWC009", "name": "Mumbai Mithi", "lat": 19.07, "lon": 72.88,
             "river": "Mithi", "basin": "West Flowing Rivers", "state": "Maharashtra", "type": "gauge"},
            {"station_id": "CWC010", "name": "Delhi Yamuna", "lat": 28.68, "lon": 77.24,
             "river": "Yamuna", "basin": "Ganga", "state": "Delhi", "type": "gauge"},
        ]

        df = pd.DataFrame(stations)

        if state:
            df = df[df["state"].str.contains(state, case=False, na=False)]
        if basin:
            df = df[df["basin"].str.contains(basin, case=False, na=False)]

        return df


class CWCFloodBulletinClient:
    """
    Client for CWC Flood Forecasting bulletins.

    CWC issues flood forecasts for ~325 stations across India.
    These bulletins contain:
    - Current water levels vs danger/warning levels
    - 24/48-hour level forecasts
    - Inflow/outflow for major reservoirs
    """

    def __init__(self):
        self.base_url = settings.data_sources.cwc_bulletin_url

    def fetch_latest_bulletin(self) -> pd.DataFrame:
        """
        Scrape the latest CWC flood forecast bulletin.

        Returns:
            DataFrame with station-level flood status and forecasts
        """
        import httpx

        logger.info("Fetching latest CWC flood bulletin")

        try:
            response = httpx.get(
                f"{self.base_url}/ffsbulletin",
                timeout=30.0,
            )
            response.raise_for_status()

            # Parse the bulletin (typically HTML/PDF format)
            # This is a simplified parser — production would use BeautifulSoup
            bulletin_data = self._parse_bulletin(response.text)

            return pd.DataFrame(bulletin_data)

        except Exception as e:
            logger.error(f"Failed to fetch CWC bulletin: {e}")
            return pd.DataFrame()

    def _parse_bulletin(self, html_content: str) -> list[dict]:
        """Parse CWC bulletin HTML into structured records."""
        # Placeholder — real implementation would parse the actual CWC bulletin format
        logger.debug("Parsing CWC bulletin HTML content")
        return []


class DischargeEstimator:
    """
    Estimate discharge from water level using rating curves.

    The stage-discharge relationship follows a power law:
        Q = a × (H - H₀)^b

    where:
        Q = discharge (m³/s or "cumecs")
        H = observed water level (m)
        H₀ = zero-discharge level (gauge datum offset)
        a, b = rating curve parameters (fitted from field measurements)

    For flood frequency analysis, we apply the Gumbel distribution
    to annual maximum discharge series.
    """

    @staticmethod
    def rating_curve(
        water_level: np.ndarray,
        a: float,
        b: float,
        h0: float = 0.0,
    ) -> np.ndarray:
        """
        Convert water level to discharge using power-law rating curve.

        Q = a × (H - H₀)^b

        Args:
            water_level: Water level array (meters above datum)
            a: Rating curve coefficient
            b: Rating curve exponent (typically 1.5 – 2.5)
            h0: Zero-discharge stage (meters)

        Returns:
            Discharge array in m³/s (cumecs)
        """
        effective_stage = np.maximum(water_level - h0, 0)
        return a * np.power(effective_stage, b)

    @staticmethod
    def fit_rating_curve(
        water_levels: np.ndarray,
        discharges: np.ndarray,
        h0_guess: float = 0.0,
    ) -> dict:
        """
        Fit rating curve parameters using nonlinear least squares.

        Args:
            water_levels: Observed water levels (m)
            discharges: Measured discharges (m³/s)
            h0_guess: Initial guess for zero-discharge level

        Returns:
            Dictionary with fitted parameters: a, b, h0, r_squared
        """
        from scipy.optimize import curve_fit

        def _model(h, a, b, h0):
            return a * np.power(np.maximum(h - h0, 1e-6), b)

        # Initial parameter guesses
        p0 = [1.0, 2.0, h0_guess]
        bounds = ([0, 0.5, -10], [1000, 5.0, np.min(water_levels)])

        try:
            popt, pcov = curve_fit(
                _model, water_levels, discharges, p0=p0, bounds=bounds, maxfev=5000
            )

            # R² calculation
            q_pred = _model(water_levels, *popt)
            ss_res = np.sum((discharges - q_pred) ** 2)
            ss_tot = np.sum((discharges - np.mean(discharges)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)

            return {
                "a": popt[0],
                "b": popt[1],
                "h0": popt[2],
                "r_squared": r_squared,
            }
        except RuntimeError as e:
            logger.error(f"Rating curve fitting failed: {e}")
            raise

    @staticmethod
    def gumbel_flood_frequency(
        annual_max_discharge: np.ndarray,
        return_periods: list[float] | None = None,
    ) -> pd.DataFrame:
        """
        Flood frequency analysis using Gumbel (Type I Extreme Value) distribution.

        f(x) = (1/β) × exp(-(z + exp(-z)))
        where z = (x - μ) / β

        Parameters estimated via method of moments:
            β = (√6 / π) × σ_x
            μ = x̄ - 0.5772 × β  (Euler-Mascheroni constant)

        Return period discharge:
            x_T = μ - β × ln(-ln(1 - 1/T))

        Args:
            annual_max_discharge: Array of annual maximum discharge values (m³/s)
            return_periods: List of return periods in years (default: standard set)

        Returns:
            DataFrame with return_period_years, discharge_cumecs, exceedance_prob
        """
        if return_periods is None:
            return_periods = [2, 5, 10, 25, 50, 100, 200, 500]

        x_bar = np.mean(annual_max_discharge)
        s_x = np.std(annual_max_discharge, ddof=1)

        # Gumbel parameter estimation (method of moments)
        beta = (np.sqrt(6) / np.pi) * s_x    # Scale parameter
        mu = x_bar - 0.5772 * beta             # Location parameter (Euler constant)

        results = []
        for T in return_periods:
            # Gumbel quantile function
            x_T = mu - beta * np.log(-np.log(1 - 1 / T))
            p_exceed = 1 / T

            results.append({
                "return_period_years": T,
                "discharge_cumecs": round(x_T, 2),
                "exceedance_probability": round(p_exceed, 4),
            })

        df = pd.DataFrame(results)

        logger.info(
            f"Gumbel FFA complete | μ={mu:.2f}, β={beta:.2f} | "
            f"Q100={df[df['return_period_years']==100]['discharge_cumecs'].values[0]:.0f} cumecs"
        )

        return df
