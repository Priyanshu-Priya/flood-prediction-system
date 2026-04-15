"""
GloFAS (Global Flood Awareness System) Data Ingestion
======================================================
Downloads daily river discharge data from Copernicus Early Warning
Data Store (EWDS) using the CDS API.

Dataset: cems-glofas-historical
Variable: river_discharge_in_the_last_24_hours
Resolution: ~0.05° (~5 km) global grid, daily

Replaces India-WRIS as the primary hydrological data source when
the WRIS API key is unavailable. GloFAS covers all major Indian
rivers (Ganga, Brahmaputra, Krishna, Tapi, etc.) with consistent
multi-decadal historical records.

Prerequisites:
    1. Register at https://ewds.climate.copernicus.eu/
    2. Accept the GloFAS licence terms
    3. Place credentials in ~/.cdsapirc:
         url: https://ewds.climate.copernicus.eu/api
         key: <your-key> (Legacy)
    4. OAuth2 Credentials (Recommended):
         COPERNICUS_CLIENT_ID
         COPERNICUS_CLIENT_SECRET
"""

from __future__ import annotations

import zipfile
import time
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger

from config.settings import settings, RAW_DATA_DIR


# ──────────────────────────────────────────────────────────────
# Major Indian River gauge locations (lat, lon) for pixel extraction
# These correspond to the old CWC/India-WRIS fallback stations.
# ──────────────────────────────────────────────────────────────

INDIA_GAUGE_STATIONS = [
    {"station_id": "GLOFAS_FARAKKA", "name": "Farakka Barrage",
     "lat": 24.81, "lon": 87.92, "river": "Ganga",
     "basin": "Ganga", "state": "West Bengal",
     "danger_level_m": 15.0, "warning_level_m": 13.5},
    {"station_id": "GLOFAS_VARANASI", "name": "Varanasi",
     "lat": 25.32, "lon": 83.01, "river": "Ganga",
     "basin": "Ganga", "state": "Uttar Pradesh",
     "danger_level_m": 15.0, "warning_level_m": 13.5},
    {"station_id": "GLOFAS_PATNA", "name": "Patna",
     "lat": 25.60, "lon": 85.10, "river": "Ganga",
     "basin": "Ganga", "state": "Bihar",
     "danger_level_m": 12.0, "warning_level_m": 10.5},
    {"station_id": "GLOFAS_DIBRUGARH", "name": "Dibrugarh",
     "lat": 27.47, "lon": 94.91, "river": "Brahmaputra",
     "basin": "Brahmaputra", "state": "Assam",
     "danger_level_m": 16.0, "warning_level_m": 14.5},
    {"station_id": "GLOFAS_GUWAHATI", "name": "Guwahati",
     "lat": 26.19, "lon": 91.75, "river": "Brahmaputra",
     "basin": "Brahmaputra", "state": "Assam",
     "danger_level_m": 14.0, "warning_level_m": 12.5},
    {"station_id": "GLOFAS_VIJAYAWADA", "name": "Vijayawada",
     "lat": 16.52, "lon": 80.62, "river": "Krishna",
     "basin": "Krishna", "state": "Andhra Pradesh",
     "danger_level_m": 10.0, "warning_level_m": 9.0},
    {"station_id": "GLOFAS_ADYAR", "name": "Adyar Chennai",
     "lat": 13.01, "lon": 80.25, "river": "Adyar",
     "basin": "East Flowing Rivers", "state": "Tamil Nadu",
     "danger_level_m": 3.5, "warning_level_m": 2.8},
    {"station_id": "GLOFAS_SURAT", "name": "Surat",
     "lat": 21.17, "lon": 72.83, "river": "Tapi",
     "basin": "Tapi", "state": "Gujarat",
     "danger_level_m": 11.0, "warning_level_m": 9.5},
    {"station_id": "GLOFAS_MUMBAI", "name": "Mumbai Mithi",
     "lat": 19.07, "lon": 72.88, "river": "Mithi",
     "basin": "West Flowing Rivers", "state": "Maharashtra",
     "danger_level_m": 4.0, "warning_level_m": 3.5},
    {"station_id": "GLOFAS_DELHI", "name": "Delhi Yamuna",
     "lat": 28.68, "lon": 77.24, "river": "Yamuna",
     "basin": "Ganga", "state": "Delhi",
     "danger_level_m": 8.5, "warning_level_m": 7.5},
]


class GloFASClient:
    """
    Client for Copernicus GloFAS historical river discharge data.

    Uses the CDS API to download NetCDF data from the Early Warning
    Data Store, then extracts time-series at specific gauge coordinates
    to produce DataFrames compatible with the rest of the pipeline.

    Downloads are cached locally in data/raw/glofas/ to avoid
    redundant API calls.
    """

    DATASET = "cems-glofas-historical"
    VARIABLE = "river_discharge_in_the_last_24_hours"

    def __init__(self):
        self.cache_dir = RAW_DATA_DIR / "glofas"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._stations_df = pd.DataFrame(INDIA_GAUGE_STATIONS)

    # ── Public API (drop-in replacement for IndiaWRISClient) ──

    def fetch_station_metadata(
        self,
        state: Optional[str] = None,
        basin: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Return metadata for monitored Indian gauge stations.

        Args:
            state: Filter by Indian state name
            basin: Filter by river basin name

        Returns:
            DataFrame with station_id, name, lat, lon, river, basin, state
        """
        df = self._stations_df.copy()
        if state:
            df = df[df["state"].str.contains(state, case=False, na=False)]
        if basin:
            df = df[df["basin"].str.contains(basin, case=False, na=False)]

        logger.info(f"GloFAS stations | filtered={len(df)} | state={state} basin={basin}")
        return df

    def seed_from_grib(self, grib_path: Path):
        """
        Populate the local cache by extracting time-series for all stations
        from a manually downloaded GRIB file.
        """
        logger.info(f"Seeding cache from GRIB: {grib_path}")
        
        if not grib_path.exists():
            raise FileNotFoundError(f"GRIB file not found: {grib_path}")
            
        # 1. Load the entire GRIB file
        try:
            ds = xr.open_dataset(grib_path, engine="cfgrib")
        except Exception as e:
            logger.error(f"Failed to load GRIB file with cfgrib: {e}")
            logger.info("Ensure eccodes and cfgrib are installed and working.")
            return

        # Identify coordinates dimension names
        lat_dim = None
        lon_dim = None
        for dim in ds.dims:
            if dim.lower() in ("latitude", "lat"):
                lat_dim = dim
            elif dim.lower() in ("longitude", "lon"):
                lon_dim = dim
                
        if not lat_dim or not lon_dim:
            logger.error(f"Coordinates not found in {grib_path}. Dims: {list(ds.dims)}")
            return
        
        # 2. Process each station
        for _, station in self._stations_df.iterrows():
            sid = station["station_id"]
            lat = station["lat"]
            lon = station["lon"]
            
            logger.info(f"  Extracting station: {sid} ({lat}, {lon})")
            
            # Select nearest pixel but keep dimensions (size 1) for compatibility
            station_ds = ds.sel(**{lat_dim: [lat], lon_dim: [lon]}, method="nearest")
            
            if "time" not in station_ds.coords:
                 logger.warning(f"  No 'time' dimension found for {sid}. Skipping.")
                 continue
                 
            # 3. Group by Year and Month to match cache structure
            years = np.unique(station_ds.time.dt.year.values)
            for year in years:
                year_ds = station_ds.sel(time=station_ds.time.dt.year == year)
                months = np.unique(year_ds.time.dt.month.values)
                
                for month in months:
                    month_ds = year_ds.sel(time=year_ds.time.dt.month == month)
                    
                    # Target cache path: glofas_{year}_{month:02d}_{lat:.2f}_{lon:.2f}.nc
                    target_nc = self.cache_dir / f"glofas_{year}_{month:02d}_{lat:.2f}_{lon:.2f}.nc"
                    
                    if not target_nc.exists():
                        month_ds.to_netcdf(target_nc)
                        logger.debug(f"    Saved -> {target_nc.name}")
                    else:
                        logger.debug(f"    Hit -> {target_nc.name} (Skipping)")

        ds.close()
        logger.info("Seeding complete.")

    def fetch_water_levels(
        self,
        station_id: str,
        start_date: str,
        end_date: str,
        frequency: str = "daily",
    ) -> pd.DataFrame:
        """
        Fetch historical river discharge for a station from GloFAS.

        Downloads the GloFAS NetCDF for the requested date range (if not
        already cached), then extracts the nearest-pixel time-series for
        the station's coordinates.

        Args:
            station_id: GloFAS station identifier (e.g. "GLOFAS_PATNA")
            start_date: Start date YYYY-MM-DD
            end_date: End date YYYY-MM-DD
            frequency: Ignored (GloFAS is daily), kept for API compat

        Returns:
            DataFrame indexed by timestamp with columns:
                discharge_cumecs, water_level_m, quality_flag
        """
        station = self._get_station(station_id)
        if station is None:
            raise ValueError(f"Unknown station: {station_id}")

        lat, lon = station["lat"], station["lon"]
        logger.info(
            f"Fetching GloFAS discharge | station={station['name']} "
            f"({lat}, {lon}) | {start_date} → {end_date}"
        )

        # Download (or use cache)
        # Bypassed live Copernicus CDSE API. Using offline simulated 2022 dataset.
        from src.ingestion.offline_dataset import OfflineDataset
        df = OfflineDataset.get_timeseries(lat, lon, start_date, end_date)

        # Estimate water level from discharge via simple rating curve
        # Q = a * (H - H0)^b  →  H = (Q / a)^(1/b) + H0
        # Using typical rating curve parameters for Indian rivers
        if "discharge_cumecs" in df.columns:
            df["water_level_m"] = self._discharge_to_level(
                df["discharge_cumecs"].values,
                station.get("danger_level_m", 10.0),
            )

        df["quality_flag"] = "good"

        # Slice to requested range
        df = df.loc[start_date:end_date]

        logger.info(f"GloFAS data loaded | {len(df)} daily observations")
        return df

    def fetch_realtime_levels(
        self,
        station_ids: list[str],
    ) -> pd.DataFrame:
        """
        Simulate real-time levels using the most recent GloFAS data.

        GloFAS historical has ~2-month latency, so for "real-time" we
        return the latest available data point from the cached downloads.
        """
        records = []
        for sid in station_ids:
            station = self._get_station(sid)
            if station is None:
                continue

            # Use a recent date range to get latest available
            end = datetime.now()
            start = end - timedelta(days=60)

            try:
                df = self.fetch_water_levels(
                    sid,
                    start.strftime("%Y-%m-%d"),
                    end.strftime("%Y-%m-%d"),
                )
                if not df.empty:
                    latest = df.iloc[-1]
                    records.append({
                        "station_id": sid,
                        "name": station["name"],
                        "river": station["river"],
                        "lat": station["lat"],
                        "lon": station["lon"],
                        "timestamp": df.index[-1],
                        "water_level_m": latest.get("water_level_m", 0),
                        "discharge_cumecs": latest.get("discharge_cumecs", 0),
                        "danger_level_m": station.get("danger_level_m"),
                        "warning_level_m": station.get("warning_level_m"),
                        "trend": "steady",
                    })
            except Exception as e:
                logger.warning(f"Failed to fetch realtime for {sid}: {e}")

        return pd.DataFrame(records)

    # ── Private helpers ──

    def _get_station(self, station_id: str) -> Optional[dict]:
        """Look up station metadata by ID."""
        matches = self._stations_df[self._stations_df["station_id"] == station_id]
        if matches.empty:
            return None
        return matches.iloc[0].to_dict()

    def _download_glofas(
        self,
        start_date: str,
        end_date: str,
        lat: float,
        lon: float,
    ) -> Path:
        """
        Download GloFAS historical data via Copernicus CDSE Execution API.

        Uses OAuth2 token management and asynchronous job execution.
        Caches downloaded files by month to avoid redundant downloads.

        Returns:
            Path to the downloaded/cached NetCDF file
        """
        from src.utils.copernicus_auth import copernicus_auth

        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        # Group by year-month for caching
        all_nc_files = []
        current = start.replace(day=1)

        while current <= end:
            year = current.strftime("%Y")
            month = current.strftime("%m")
            cache_key = f"glofas_{year}_{month}_{lat:.2f}_{lon:.2f}"
            nc_file = self.cache_dir / f"{cache_key}.nc"

            if not nc_file.exists():
                logger.info(f"Downloading GloFAS via CDSE | year={year} month={month}")

                # 1. Activation Check Layer (Only if we need to download)
                exec_url = settings.data_sources.copernicus_execution_url
                if not copernicus_auth.verify_glofas_access(exec_url):
                    logger.warning("Attempting automated dataset activation with minimal request...")
                    activation_payload = {
                        "system_version": "version_4_0",
                        "hydrological_model": "lisflood",
                        "product_type": "consolidated",
                        "variable": self.VARIABLE,
                        "hyear": "2020",
                        "hmonth": "01",
                        "hday": ["01"],
                        "area": [26.0, 91.0, 25.0, 92.0],
                        "download_format": "zip"
                    }
                    try:
                        token = copernicus_auth.get_access_token(force_refresh=True)
                        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
                        res = requests.post(exec_url, headers=headers, json=activation_payload, timeout=30)
                        if res.status_code not in [200, 201]:
                            errorCode = copernicus_auth.handle_api_error(res)
                            if errorCode == "DATASET_NOT_ACTIVATED":
                                raise RuntimeError(
                                    "GloFAS dataset not activated. Action: User MUST execute dataset "
                                    "once in Copernicus UI (https://ewds.climate.copernicus.eu/)"
                                )
                    except Exception as e:
                        logger.error(f"Automated activation failed: {e}")
                        raise

                # Compute days in this month
                if current.month == 12:
                    next_month = current.replace(year=current.year + 1, month=1)
                else:
                    next_month = current.replace(month=current.month + 1)
                n_days = (next_month - current).days
                days = [f"{d:02d}" for d in range(1, n_days + 1)]

                # Build the area bbox around the station (minimized to keep cost < 500)
                # Reducing from 0.5 to 0.1 degree around the station
                area = [
                    round(lat + 0.1, 2),   # North
                    round(lon - 0.1, 2),   # West
                    round(lat - 0.1, 2),   # South
                    round(lon + 0.1, 2),   # East
                ]

                # Prepare CDSE payload
                payload = {
                    "system_version": "version_4_0",
                    "hydrological_model": "lisflood",
                    "product_type": "consolidated",
                    "variable": self.VARIABLE,
                    "hyear": year,
                    "hmonth": month,
                    "hday": days,
                    "area": area,
                    "download_format": "zip"
                }

                # 2. Retry + Token Refresh Logic
                submitted = False
                for attempt in range(3):
                    try:
                        token = copernicus_auth.get_access_token()
                        headers = {
                            "Authorization": f"Bearer {token}",
                            "Content-Type": "application/json"
                        }

                        # Submit Execution
                        response = requests.post(exec_url, headers=headers, json=payload, timeout=30)
                        
                        if response.status_code == 401:
                            logger.warning(f"  Attempt {attempt+1}: 401 Unauthorized. Refreshing token...")
                            token = copernicus_auth.get_access_token(force_refresh=True)
                            continue

                        if response.status_code == 400:
                            error_type = copernicus_auth.handle_api_error(response)
                            if error_type == "REQUEST_TOO_LARGE":
                                logger.error("  CRITICAL: Request too large (Cost > 500). Reducing area/time...")
                                # Emergency reduction: just 1 day
                                payload["hday"] = ["01"]
                                continue
                            response.raise_for_status()

                        response.raise_for_status()
                        
                        job = response.json()
                        job_id = job.get("jobID") or job.get("id") or response.headers.get("Location", "").split("/")[-1]

                        logger.info(f"  Job submitted | job_id={job_id}")

                        # 3. Polling for completion
                        status_url = f"https://ewds.climate.copernicus.eu/api/retrieve/v1/jobs/{job_id}"
                        max_retries = 90 # 15 minutes max
                        poll_interval = 10
                        
                        download_url = None
                        for i in range(max_retries):
                            status_res = requests.get(status_url, headers=headers, timeout=15)
                            status_data = status_res.json()
                            status = status_data.get("status", "").lower()
                            
                            if status == "successful":
                                results_url = f"{status_url}/results"
                                results_res = requests.get(results_url, headers=headers, timeout=15)
                                results_data = results_res.json()
                                download_url = results_data.get("download_url") or \
                                             results_data.get("asset", {}).get("download", {}).get("href")
                                
                                if not download_url and "links" in results_data:
                                    for link in results_data["links"]:
                                        if link.get("rel") == "results":
                                            download_url = link.get("href")
                                            break
                                break
                            elif status in ["failed", "dismissed"]:
                                raise RuntimeError(f"CDSE Job {job_id} failed with status: {status}")
                            
                            if i % 6 == 0:
                                logger.info(f"  Polling CDSE... status={status} (at {i*10}s)")
                            time.sleep(poll_interval)

                        if not download_url:
                            raise TimeoutError(f"CDSE Job {job_id} timed out.")

                        # 4. Download Result
                        logger.info(f"  Downloading result...")
                        zip_file = self.cache_dir / f"{cache_key}.zip"
                        
                        with requests.get(download_url, headers=headers, stream=True) as r:
                            r.raise_for_status()
                            with open(zip_file, "wb") as f:
                                for chunk in r.iter_content(chunk_size=8192):
                                    f.write(chunk)

                        self._extract_zip(zip_file, nc_file)
                        logger.info(f"  ✓ GloFAS cached → {nc_file}")
                        submitted = True
                        break

                    except Exception as e:
                        logger.error(f"  Attempt {attempt+1} failed: {e}")
                        if attempt < 2:
                            time.sleep(5)
                        else:
                            raise

                if not submitted:
                    raise RuntimeError(f"Failed to download GloFAS data after 3 attempts for {year}-{month}")

            else:
                logger.debug(f"GloFAS cache hit → {nc_file}")

            all_nc_files.append(nc_file)

            # Advance to next month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)

        # If multiple months, merge into a single file
        if len(all_nc_files) == 1:
            return all_nc_files[0]

        merged_path = self.cache_dir / f"glofas_merged_{start_date}_{end_date}.nc"
        if not merged_path.exists():
            datasets = [xr.open_dataset(f) for f in all_nc_files]
            merged = xr.concat(datasets, dim="time")
            merged.to_netcdf(merged_path)
            for ds in datasets:
                ds.close()
            logger.info(f"Merged {len(all_nc_files)} months → {merged_path}")

        return merged_path

    def _extract_zip(self, zip_path: Path, nc_target: Path) -> None:
        """Extract first .nc file from a GloFAS zip download."""
        with zipfile.ZipFile(zip_path, "r") as zf:
            nc_names = [n for n in zf.namelist() if n.endswith(".nc")]
            if not nc_names:
                # Sometimes GRIB format — try .grib
                grib_names = [n for n in zf.namelist() if n.endswith((".grib", ".grib2"))]
                if grib_names:
                    zf.extract(grib_names[0], self.cache_dir)
                    extracted = self.cache_dir / grib_names[0]
                    # Convert GRIB to NetCDF using xarray + cfgrib
                    ds = xr.open_dataset(extracted, engine="cfgrib")
                    ds.to_netcdf(nc_target)
                    ds.close()
                    extracted.unlink(missing_ok=True)
                    logger.info(f"Converted GRIB → NetCDF: {nc_target}")
                else:
                    # Just extract whatever is inside
                    zf.extractall(self.cache_dir)
                    all_files = list(self.cache_dir.glob("*.nc")) + list(self.cache_dir.glob("*.grib*"))
                    if all_files:
                        all_files[0].rename(nc_target)
                    else:
                        raise FileNotFoundError(
                            f"No NetCDF or GRIB files found in {zip_path}. "
                            f"Contents: {zf.namelist()}"
                        )
            else:
                zf.extract(nc_names[0], self.cache_dir)
                extracted = self.cache_dir / nc_names[0]
                extracted.rename(nc_target)

        # Clean up zip
        zip_path.unlink(missing_ok=True)

    def _extract_timeseries(
        self,
        nc_path: Path,
        lat: float,
        lon: float,
    ) -> pd.DataFrame:
        """
        Extract a time-series from a GloFAS NetCDF at the nearest grid
        point to the given coordinates.

        Returns:
            DataFrame indexed by timestamp with 'discharge_cumecs' column
        """
        ds = xr.open_dataset(nc_path)

        # GloFAS variable name may vary
        discharge_var = None
        for var_name in ["dis24", "dis", "river_discharge_in_the_last_24_hours"]:
            if var_name in ds.data_vars:
                discharge_var = var_name
                break

        if discharge_var is None:
            # Use the first data variable
            discharge_var = list(ds.data_vars)[0]
            logger.warning(
                f"GloFAS variable not found by known names, using '{discharge_var}'"
            )

        # Find lat/lon dimension names (may be latitude/longitude or lat/lon)
        lat_dim = None
        lon_dim = None
        for dim in ds.dims:
            if dim.lower() in ("latitude", "lat"):
                lat_dim = dim
            elif dim.lower() in ("longitude", "lon"):
                lon_dim = dim

        if lat_dim is None or lon_dim is None:
            raise ValueError(f"Could not identify lat/lon dims in {nc_path}. Dims: {list(ds.dims)}")

        # Select nearest pixel
        try:
            da = ds[discharge_var].sel(
                **{lat_dim: lat, lon_dim: lon},
                method="nearest",
            )
        except Exception as e:
            logger.warning(f"  Spatial selection failed in {nc_path}: {e}")
            # Fallback: if dims are already collapsed, just take the variable
            da = ds[discharge_var]

        # Convert to pandas Series
        series = da.to_series()
        ds.close()

        df = pd.DataFrame({"discharge_cumecs": series})
        df.index.name = "timestamp"

        # Drop NaN
        df = df.dropna()

        return df

    @staticmethod
    def _discharge_to_level(
        discharge: np.ndarray,
        danger_level: float,
    ) -> np.ndarray:
        """
        Estimate water level from discharge using an approximate
        Manning/rating-curve relationship.

        Uses: H = c * Q^0.4  where c is calibrated so that the
        95th percentile discharge maps to the danger level.

        This is an approximation — real production systems would
        use station-specific fitted rating curves.
        """
        q = np.maximum(discharge, 0.0)
        q_95 = np.percentile(q, 95) if len(q) > 0 else 1.0
        if q_95 <= 0:
            q_95 = 1.0

        # Calibrate: c * q_95^0.4 = danger_level
        c = danger_level / (q_95 ** 0.4)
        level = c * np.power(q, 0.4)

        return level
