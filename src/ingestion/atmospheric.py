"""
Atmospheric / Rainfall Data Ingestion — India Edition
======================================================
Integrates:
  - IMD (India Meteorological Department) gridded rainfall (0.25° × 0.25°)
  - NASA GPM (Global Precipitation Measurement) — 30-min, 0.1° resolution
  - ERA5 reanalysis (temperature, soil moisture, surface runoff)

The hydro-meteorological link:
  Rainfall → Infiltration → Runoff → Flood
  The antecedent soil moisture determines how much rain becomes runoff.
  Saturated soil = almost all rain becomes surface flow.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger

from config.settings import settings, RAW_DATA_DIR


class IMDRainfallLoader:
    """
    Load IMD (India Meteorological Department) gridded rainfall data.

    IMD provides daily gridded rainfall at 0.25° × 0.25° resolution
    covering all of India. The data spans from 1901 to near-present.

    Format: Binary (.grd) files with fixed grid dimensions
    Grid: 31°N–7°N latitude, 67°E–100°E longitude
    Resolution: 0.25° (≈ 28 km at Indian latitudes)

    This data is the gold standard for understanding rainfall patterns
    across India, as it's derived from a dense gauge network.
    """

    # IMD grid specifications
    N_LAT = 129   # Number of latitude points (6.5°N to 38.5°N)
    N_LON = 135   # Number of longitude points (66.5°E to 100.0°E)
    LAT_MIN = 6.5
    LAT_MAX = 38.5
    LON_MIN = 66.5
    LON_MAX = 100.0
    RESOLUTION = 0.25

    def __init__(self):
        self.data_dir = RAW_DATA_DIR / "imd"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load_daily_rainfall(
        self,
        year: int,
        filepath: Optional[Path] = None,
    ) -> xr.DataArray:
        """
        Load IMD daily gridded rainfall data for a given year.

        The .grd files are binary (float32), arranged as:
        [day1_lat1_lon1, day1_lat1_lon2, ..., day1_latN_lonM, day2_..., ...]

        Args:
            year: Year to load (e.g., 2023)
            filepath: Path to the .grd binary file (if not in default location)

        Returns:
            xarray DataArray with dims (time, latitude, longitude), units mm/day
        """
        if filepath is None:
            filepath = self.data_dir / f"Rainfall_ind{year}_rfp25.grd"

        if not filepath.exists():
            logger.warning(f"IMD rainfall file not found: {filepath}")
            raise FileNotFoundError(
                f"Download IMD data from {settings.data_sources.imd_base_url} "
                f"and place at {filepath}"
            )

        logger.info(f"Loading IMD daily rainfall for {year} from {filepath}")

        # Determine number of days in year
        is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
        n_days = 366 if is_leap else 365

        # Read binary data
        data = np.fromfile(str(filepath), dtype=np.float32)
        expected_size = n_days * self.N_LAT * self.N_LON

        if data.size != expected_size:
            raise ValueError(
                f"Data size mismatch: got {data.size}, expected {expected_size} "
                f"({n_days} days × {self.N_LAT} lats × {self.N_LON} lons)"
            )

        # Reshape: (days, lat, lon)
        data = data.reshape(n_days, self.N_LAT, self.N_LON)

        # Replace missing values (-999.0 or 99.9 in IMD convention)
        data[data < 0] = np.nan
        data[data > 500] = np.nan  # Physical upper bound sanity check

        # Create coordinate arrays
        lats = np.arange(self.LAT_MIN, self.LAT_MAX + self.RESOLUTION / 2, self.RESOLUTION)
        lons = np.arange(self.LON_MIN, self.LON_MAX + self.RESOLUTION / 2, self.RESOLUTION)
        times = pd.date_range(start=f"{year}-01-01", periods=n_days, freq="D")

        # Truncate coords to match data shape exactly
        lats = lats[:self.N_LAT]
        lons = lons[:self.N_LON]

        da = xr.DataArray(
            data,
            dims=["time", "latitude", "longitude"],
            coords={
                "time": times,
                "latitude": lats,
                "longitude": lons,
            },
            attrs={
                "units": "mm/day",
                "source": "IMD Gridded Rainfall",
                "resolution": "0.25 degree",
                "long_name": "Daily Rainfall",
            },
        )

        logger.info(
            f"Loaded IMD rainfall | shape={da.shape} | "
            f"mean={float(da.mean()):.1f} mm/day"
        )

        return da

    def extract_for_aoi(
        self,
        rainfall: xr.DataArray,
        bbox: tuple[float, float, float, float],
    ) -> xr.DataArray:
        """
        Extract rainfall data for a specific Area of Interest.

        Args:
            rainfall: Full India rainfall DataArray
            bbox: (min_lon, min_lat, max_lon, max_lat) in WGS84

        Returns:
            Clipped DataArray for the AOI
        """
        min_lon, min_lat, max_lon, max_lat = bbox

        clipped = rainfall.sel(
            latitude=slice(min_lat, max_lat),
            longitude=slice(min_lon, max_lon),
        )

        logger.info(f"Extracted AOI rainfall | shape={clipped.shape}")
        return clipped


class GPMPrecipitationLoader:
    """
    Load NASA GPM (Global Precipitation Measurement) IMERG data.

    GPM IMERG provides near-global precipitation at:
    - Spatial: 0.1° × 0.1° (≈ 11 km)
    - Temporal: 30-minute intervals (half-hourly)

    Products:
    - Early (4-hour latency) — real-time monitoring
    - Late (14-hour latency) — better gauge correction
    - Final (3.5-month latency) — research quality

    This is superior to IMD for:
    1. Sub-daily temporal resolution (flash flood detection)
    2. Real-time availability (within hours)
    3. Oceanic coverage (coastal flooding events)
    """

    def __init__(self):
        self.base_url = settings.data_sources.gpm_opendap_url
        self.token = settings.data_sources.nasa_earthdata_token

    def load_imerg_halfhourly(
        self,
        bbox: tuple[float, float, float, float],
        start_date: str,
        end_date: str,
        product: str = "Late",  # "Early", "Late", or "Final"
    ) -> xr.Dataset:
        """
        Load GPM IMERG half-hourly precipitation via OPeNDAP.

        Args:
            bbox: (min_lon, min_lat, max_lon, max_lat)
            start_date: Start date (ISO format)
            end_date: End date (ISO format)
            product: IMERG product type

        Returns:
            xarray Dataset with precipitation rate (mm/hr) and quality index
        """
        logger.info(
            f"Loading GPM IMERG {product} | bbox={bbox} | "
            f"dates={start_date}/{end_date}"
        )

        # Construct OPeNDAP URL for IMERG
        product_map = {
            "Early": "GPM_3IMERGHHE",
            "Late": "GPM_3IMERGHHL",
            "Final": "GPM_3IMERGHH",
        }

        try:
            ds = xr.open_dataset(
                f"{self.base_url}/{product_map[product]}.07",
                engine="netcdf4",
            )

            # Slice to AOI and time range
            min_lon, min_lat, max_lon, max_lat = bbox
            ds = ds.sel(
                lat=slice(min_lat, max_lat),
                lon=slice(min_lon, max_lon),
                time=slice(start_date, end_date),
            )

            logger.info(f"Loaded GPM IMERG | shape={dict(ds.dims)}")
            return ds

        except Exception as e:
            logger.error(f"GPM OPeNDAP load failed: {e}")
            logger.info("Falling back to local GPM HDF5 files...")
            return self._load_local_hdf5(bbox, start_date, end_date)

    def _load_local_hdf5(
        self,
        bbox: tuple[float, float, float, float],
        start_date: str,
        end_date: str,
    ) -> xr.Dataset:
        """
        Load GPM data from pre-downloaded HDF5 files.
        Files should be in data/raw/gpm/ directory.
        """
        gpm_dir = RAW_DATA_DIR / "gpm"
        if not gpm_dir.exists():
            raise FileNotFoundError(
                f"GPM data directory not found: {gpm_dir}. "
                "Download IMERG HDF5 files from https://disc.gsfc.nasa.gov/"
            )

        import h5py

        hdf_files = sorted(gpm_dir.glob("*.HDF5"))
        logger.info(f"Found {len(hdf_files)} local GPM HDF5 files")

        datasets = []
        for f in hdf_files:
            try:
                with h5py.File(f, "r") as h5:
                    precip = h5["Grid/precipitation"][:]
                    lats = h5["Grid/lat"][:]
                    lons = h5["Grid/lon"][:]

                    da = xr.DataArray(
                        precip,
                        dims=["lon", "lat"],
                        coords={"lat": lats, "lon": lons},
                    )
                    datasets.append(da)
            except Exception as e:
                logger.warning(f"Failed to read {f}: {e}")

        if not datasets:
            raise RuntimeError("No valid GPM files could be loaded")

        return xr.concat(datasets, dim="time").to_dataset(name="precipitation")


class ERA5Loader:
    """
    Load ERA5 reanalysis data via CDS API.

    ERA5 provides hourly global reanalysis at 0.25° resolution.
    Key variables for flood modeling:
    - Total precipitation (tp)
    - Volumetric soil water layers 1-4 (swvl1-swvl4)
    - 2m temperature (t2m)
    - Surface runoff (sro)
    - Evaporation (e)
    - Snowmelt (smlt) — relevant for Himalayan basins
    """

    # ERA5 variables relevant to flood modeling
    FLOOD_VARIABLES = {
        "total_precipitation": "tp",
        "soil_moisture_l1": "swvl1",    # 0-7 cm depth
        "soil_moisture_l2": "swvl2",    # 7-28 cm depth
        "soil_moisture_l3": "swvl3",    # 28-100 cm depth
        "soil_moisture_l4": "swvl4",    # 100-289 cm depth
        "2m_temperature": "t2m",
        "surface_runoff": "sro",
        "evaporation": "e",
        "snowmelt": "smlt",
        "10m_u_wind": "u10",
        "10m_v_wind": "v10",
    }

    def __init__(self):
        self.data_dir = RAW_DATA_DIR / "era5"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_era5(
        self,
        bbox: tuple[float, float, float, float],
        start_date: str,
        end_date: str,
        variables: list[str] | None = None,
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Download ERA5 data via CDS API (Copernicus Climate Data Store).

        Requires:
        1. CDS API key in ~/.cdsapirc
        2. Accept license at https://cds.climate.copernicus.eu/

        Args:
            bbox: (min_lon, min_lat, max_lon, max_lat) in WGS84
            start_date: Start date (YYYY-MM-DD)
            end_date: End date
            variables: List of ERA5 variable names (default: flood-relevant set)
            output_path: Where to save the NetCDF file

        Returns:
            Path to downloaded NetCDF file
        """
        import cdsapi

        if variables is None:
            variables = list(self.FLOOD_VARIABLES.keys())

        if output_path is None:
            output_path = self.data_dir / f"era5_{start_date}_{end_date}.nc"

        if output_path.exists():
            logger.info(f"ERA5 data already exists: {output_path}")
            return output_path

        min_lon, min_lat, max_lon, max_lat = bbox

        logger.info(
            f"Downloading ERA5 | vars={len(variables)} | "
            f"bbox={bbox} | dates={start_date}/{end_date}"
        )

        c = cdsapi.Client()

        # Parse date range into years/months/days
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        c.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "format": "netcdf",
                "variable": variables,
                "year": [str(y) for y in range(start.year, end.year + 1)],
                "month": [f"{m:02d}" for m in range(1, 13)],
                "day": [f"{d:02d}" for d in range(1, 32)],
                "time": [f"{h:02d}:00" for h in range(24)],
                "area": [max_lat, min_lon, min_lat, max_lon],  # N, W, S, E
            },
            str(output_path),
        )

        logger.info(f"ERA5 download complete: {output_path}")
        return output_path

    def load_era5(
        self,
        filepath: Path,
        bbox: Optional[tuple[float, float, float, float]] = None,
    ) -> xr.Dataset:
        """
        Load ERA5 NetCDF file into xarray Dataset.

        Args:
            filepath: Path to ERA5 NetCDF file
            bbox: Optional spatial subset

        Returns:
            xarray Dataset with standardized variable names
        """
        logger.info(f"Loading ERA5 from {filepath}")

        ds = xr.open_dataset(filepath, chunks={"time": 24})

        if bbox:
            min_lon, min_lat, max_lon, max_lat = bbox
            ds = ds.sel(
                latitude=slice(max_lat, min_lat),  # ERA5 lat is descending
                longitude=slice(min_lon, max_lon),
            )

        logger.info(f"ERA5 loaded | dims={dict(ds.dims)} | vars={list(ds.data_vars)}")
        return ds

    def compute_daily_aggregates(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Aggregate hourly ERA5 data to daily statistics.

        Precipitation: daily sum
        Soil moisture: daily mean
        Temperature: daily mean, min, max
        Runoff: daily sum
        """
        logger.info("Computing daily aggregates from hourly ERA5")

        daily = xr.Dataset()

        # Precipitation — daily cumulative (ERA5 stores hourly in meters)
        if "tp" in ds:
            daily["precip_daily_mm"] = ds["tp"].resample(time="1D").sum() * 1000

        # Soil moisture — daily mean (volumetric, m³/m³)
        for layer in ["swvl1", "swvl2", "swvl3", "swvl4"]:
            if layer in ds:
                daily[f"{layer}_mean"] = ds[layer].resample(time="1D").mean()

        # Temperature — daily stats (Kelvin → Celsius)
        if "t2m" in ds:
            t2m_c = ds["t2m"] - 273.15
            daily["temp_mean_c"] = t2m_c.resample(time="1D").mean()
            daily["temp_min_c"] = t2m_c.resample(time="1D").min()
            daily["temp_max_c"] = t2m_c.resample(time="1D").max()

        # Surface runoff — daily sum (meters → mm)
        if "sro" in ds:
            daily["runoff_daily_mm"] = ds["sro"].resample(time="1D").sum() * 1000

        # Snowmelt — daily sum (relevant for Himalayan basins)
        if "smlt" in ds:
            daily["snowmelt_daily_mm"] = ds["smlt"].resample(time="1D").sum() * 1000

        return daily
