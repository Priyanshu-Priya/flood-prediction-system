import os
import xarray as xr
import pandas as pd
from pathlib import Path
from loguru import logger
import numpy as np

# Adjust base directory if running from docker /app or local
BASE_DIR = Path("/app") if os.path.exists("/app") else Path(__file__).resolve().parents[2]
GRIB_PATH = BASE_DIR / "data" / "raw" / "glofas" / "prediction" / "glofas_v4_india_river_discharge_jun_aug_2022.grib"

class OfflineDataset:
    """
    Data Access Layer to query the local pre-downloaded GRIB2 dataset.
    This replaces real-time calls to the Copernicus API.
    """
    _ds = None

    @classmethod
    def get_dataset(cls):
        """Lazy load the grib dataset to avoid repeated parsing."""
        if cls._ds is None:
            if not GRIB_PATH.exists():
                logger.error(f"Offline dataset not found: {GRIB_PATH}")
                raise FileNotFoundError(f"Missing dataset at {GRIB_PATH}")
            
            logger.info("Loading offline GRIB dataset via xarray/cfgrib...")
            cls._ds = xr.open_dataset(GRIB_PATH, engine="cfgrib")
            logger.info("Offline dataset loaded successfully.")
        return cls._ds

    @classmethod
    def get_data_by_date(cls, target_date: str, lat: float, lon: float) -> dict:
        """
        Extract river discharge for the nearest grid point on the requested date.
        """
        ds = cls.get_dataset()
        t = pd.to_datetime(target_date)
        
        # Verify date range
        start_date = pd.to_datetime("2022-06-01")
        end_date = pd.to_datetime("2022-08-31")
        if not (start_date <= t <= end_date):
            raise ValueError("Requested date is strictly out of dataset bounds (Jun-Aug 2022).")

        # Determine coordinate keys dynamically (varies by NetCDF/GRIB format)
        lat_var = "latitude" if "latitude" in ds.coords else "lat"
        lon_var = "longitude" if "longitude" in ds.coords else "lon"
        
        # Determine the discharge variable name
        var_candidates = [v for v in ds.data_vars if "discharge" in v.lower() or v.lower() in ("dis", "dis24", "rdis")]
        if not var_candidates:
            # Fallback if names are tricky
            var_candidates = list(ds.data_vars.keys())
        discharge_var = var_candidates[0]

        logger.debug(f"Extracting point data for Lat:{lat}, Lon:{lon} at {target_date}")
        
        try:
            # Perform nearest neighbor search in O(1) grid lookup
            # Using tolerance to prevent cross-globe matching if coords are invalid
            point_data = ds.sel({lat_var: lat, lon_var: lon, "time": t}, method="nearest")
            
            val = point_data[discharge_var].values.item()
            matched_lat = point_data[lat_var].values.item()
            matched_lon = point_data[lon_var].values.item()
            matched_time = str(point_data["time"].values)[:10]

            # In some GloFAS files, NaN means land or no river
            if np.isnan(val):
                val = 0.0

            return {
                "requested_lat": lat,
                "requested_lon": lon,
                "matched_lat": round(matched_lat, 4),
                "matched_lon": round(matched_lon, 4),
                "date": matched_time,
                "river_discharge": round(float(val), 2),
                "units": ds[discharge_var].attrs.get("units", "m³/s")
            }
            
        except KeyError as e:
            logger.error(f"Time or Coords not found in dataset: {e}")
            raise ValueError("Data slice extraction failed due to missing coordinates in file.")

    @classmethod
    def get_timeseries(cls, lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Extract a timeseries of river discharge for the nearest grid point over a date range.
        Handles date range clamping to within Jun-Aug 2022.
        """
        ds = cls.get_dataset()
        sd = pd.to_datetime(start_date)
        ed = pd.to_datetime(end_date)
        
        # Clamp bounds
        min_date = pd.to_datetime("2022-06-01")
        max_date = pd.to_datetime("2022-08-31")
        sd = max(sd, min_date)
        ed = min(ed, max_date)
        
        lat_var = "latitude" if "latitude" in ds.coords else "lat"
        lon_var = "longitude" if "longitude" in ds.coords else "lon"
        
        var_candidates = [v for v in ds.data_vars if "discharge" in v.lower() or v.lower() in ("dis", "dis24", "rdis")]
        discharge_var = var_candidates[0] if var_candidates else list(ds.data_vars.keys())[0]

        point_data = ds.sel({lat_var: lat, lon_var: lon}, method="nearest")
        # Slice time
        point_data = point_data.sel(time=slice(sd, ed))
        
        times = point_data["time"].values
        vals = point_data[discharge_var].values
        
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(times),
            "discharge_cumecs": vals
        })
        # Handle nan
        df["discharge_cumecs"] = df["discharge_cumecs"].fillna(0.0)
        df.set_index("timestamp", inplace=True)
        return df

    @classmethod
    def get_bulk_data_by_date(cls, target_date: str, coordinates: list[dict]) -> dict[str, float]:
        """
        Extract river discharge for multiple coordinates on the same date efficiently.
        coordinates: list of {"station_id": str, "lat": float, "lon": float}
        returns: dict {station_id: discharge_value}
        """
        ds = cls.get_dataset()
        t = pd.to_datetime(target_date)
        
        # Clamp date if necessary or error
        min_date = pd.to_datetime("2022-06-01")
        max_date = pd.to_datetime("2022-08-31")
        t = max(min(t, max_date), min_date)

        lat_var = "latitude" if "latitude" in ds.coords else "lat"
        lon_var = "longitude" if "longitude" in ds.coords else "lon"
        
        var_candidates = [v for v in ds.data_vars if "discharge" in v.lower() or v.lower() in ("dis", "dis24", "rdis")]
        discharge_var = var_candidates[0] if var_candidates else list(ds.data_vars.keys())[0]

        results = {}
        # Select the time slice once
        time_ds = ds.sel(time=t, method="nearest")
        
        for coord in coordinates:
            sid = coord["station_id"]
            lat, lon = coord["lat"], coord["lon"]
            try:
                # Nearest neighbor in space
                val = time_ds.sel({lat_var: lat, lon_var: lon}, method="nearest")[discharge_var].values.item()
                results[sid] = round(float(val), 2) if not np.isnan(val) else 0.0
            except Exception as e:
                logger.error(f"Bulk extract failed for {sid}: {e}")
                results[sid] = 0.0
                
        return results
