"""
Sentinel-1 SAR Data Ingestion — STAC-based (No GEE)
=====================================================
Pulls Sentinel-1 RTC (Radiometrically Terrain Corrected) SAR data
via Microsoft Planetary Computer's STAC API.

SAR is critical for flood detection because radar penetrates clouds
— standard optical imagery is useless during storm events.

Key physics:
- Water surfaces produce specular reflection → low backscatter (dark in SAR)
- Rough surfaces (vegetation, buildings) → high backscatter (bright)
- Threshold: σ⁰_VV < -16 dB typically indicates open water
"""

from __future__ import annotations

import warnings
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import xarray as xr
from loguru import logger
from shapely.geometry import box, mapping

from config.settings import settings

# Suppress noisy STAC warnings
warnings.filterwarnings("ignore", category=UserWarning, module="stackstac")


class SentinelSARIngestion:
    """
    Fetches and preprocesses Sentinel-1 SAR imagery via STAC API.

    Uses Microsoft Planetary Computer as primary source (free, no auth required
    for unsigned access). Falls back to Copernicus Data Space if needed.

    The data is returned as analysis-ready xarray DataArrays with:
    - Radiometric terrain correction (RTC) already applied
    - VV and VH polarization bands
    - Proper CRS and spatial reference
    """

    def __init__(self):
        self.stac_url = settings.data_sources.stac_api_url
        self.collection = settings.data_sources.sentinel1_collection

    def search_scenes(
        self,
        bbox: tuple[float, float, float, float],
        start_date: str,
        end_date: str,
        max_items: int = 50,
        orbit_direction: Optional[str] = None,  # "ascending" or "descending"
    ) -> list[dict]:
        """
        Search for Sentinel-1 scenes covering the given AOI and time range.

        Args:
            bbox: (min_lon, min_lat, max_lon, max_lat) in WGS84
            start_date: ISO format start date (e.g., "2024-07-01")
            end_date: ISO format end date
            max_items: Maximum number of scenes to return
            orbit_direction: Filter by orbit direction (reduces geometric distortion)

        Returns:
            List of STAC item dictionaries with scene metadata
        """
        import pystac_client

        logger.info(
            f"Searching Sentinel-1 SAR | bbox={bbox} | "
            f"dates={start_date}/{end_date} | max={max_items}"
        )

        catalog = pystac_client.Client.open(
            self.stac_url,
            modifier=self._sign_planetary_computer,
        )

        search_params = {
            "collections": [self.collection],
            "bbox": bbox,
            "datetime": f"{start_date}/{end_date}",
            "max_items": max_items,
        }

        # Filter by orbit direction if specified
        if orbit_direction:
            search_params["query"] = {
                "sat:orbit_state": {"eq": orbit_direction}
            }

        search = catalog.search(**search_params)
        items = list(search.items())

        logger.info(f"Found {len(items)} Sentinel-1 scenes")

        return [
            {
                "id": item.id,
                "datetime": item.datetime.isoformat() if item.datetime else None,
                "geometry": item.geometry,
                "bbox": item.bbox,
                "orbit_state": item.properties.get("sat:orbit_state"),
                "relative_orbit": item.properties.get("sat:relative_orbit"),
                "assets": {k: v.href for k, v in item.assets.items()},
            }
            for item in items
        ]

    def load_sar_stack(
        self,
        bbox: tuple[float, float, float, float],
        start_date: str,
        end_date: str,
        resolution: float = 10.0,
        bands: list[str] | None = None,
        max_items: int = 30,
    ) -> xr.Dataset:
        """
        Load Sentinel-1 SAR data as a lazy xarray Dataset via stackstac.

        This creates a Dask-backed datacube that only downloads pixels
        when computation is triggered (.compute() or .values).

        Args:
            bbox: AOI bounding box in WGS84
            start_date: Start date (ISO format)
            end_date: End date (ISO format)
            resolution: Target resolution in meters (default 10m for SAR)
            bands: Polarization bands to load (default: ["vv", "vh"])
            max_items: Max scenes to stack

        Returns:
            xr.Dataset with dimensions (time, y, x) and VV/VH data variables
        """
        import pystac_client
        import stackstac

        if bands is None:
            bands = ["vv", "vh"]

        logger.info(f"Loading SAR stack | bands={bands} | resolution={resolution}m")

        catalog = pystac_client.Client.open(
            self.stac_url,
            modifier=self._sign_planetary_computer,
        )

        search = catalog.search(
            collections=[self.collection],
            bbox=bbox,
            datetime=f"{start_date}/{end_date}",
            max_items=max_items,
        )
        items = list(search.items())

        if not items:
            raise ValueError(
                f"No Sentinel-1 scenes found for bbox={bbox}, "
                f"dates={start_date}/{end_date}"
            )

        logger.info(f"Stacking {len(items)} scenes into datacube")

        # Create lazy datacube — nothing downloaded yet
        stack = stackstac.stack(
            items,
            assets=bands,
            resolution=resolution,
            bounds_latlon=bbox,
            dtype=np.float32,
            fill_value=np.nan,
            chunksize=settings.geospatial.chunk_size,
        )

        # Convert to Dataset with named variables
        ds = stack.to_dataset(dim="band")
        ds.attrs["source"] = "Sentinel-1 RTC via Planetary Computer"
        ds.attrs["units"] = "dB (backscatter coefficient σ⁰)"

        logger.info(
            f"SAR datacube created | shape={dict(ds.dims)} | "
            f"CRS={stack.crs if hasattr(stack, 'crs') else 'EPSG:32644'}"
        )

        return ds

    def create_temporal_composite(
        self,
        sar_stack: xr.Dataset,
        method: str = "median",
        window_days: int = 12,
    ) -> xr.Dataset:
        """
        Create temporal composites to reduce speckle and fill gaps.

        Sentinel-1 has a 12-day revisit time → bi-weekly composites
        provide good temporal consistency while reducing noise.

        Args:
            sar_stack: Raw SAR datacube with time dimension
            method: Aggregation method ("median", "mean", "min")
            window_days: Temporal window for compositing

        Returns:
            Composited SAR dataset with reduced speckle
        """
        logger.info(f"Creating temporal composite | method={method} | window={window_days}d")

        # Resample to regular temporal grid
        if method == "median":
            composite = sar_stack.resample(time=f"{window_days}D").median()
        elif method == "mean":
            composite = sar_stack.resample(time=f"{window_days}D").mean()
        elif method == "min":
            # Min composite enhances water detection (water = low backscatter)
            composite = sar_stack.resample(time=f"{window_days}D").min()
        else:
            raise ValueError(f"Unknown compositing method: {method}")

        composite.attrs["composite_method"] = method
        composite.attrs["composite_window_days"] = window_days

        return composite

    def compute_change_detection(
        self,
        pre_flood: xr.DataArray,
        during_flood: xr.DataArray,
    ) -> xr.DataArray:
        """
        Detect flooded areas by differencing pre-flood and during-flood SAR images.

        The change detection approach:
        1. Compute log-ratio: Δσ⁰ = σ⁰_flood - σ⁰_pre (in dB)
        2. Strong negative Δσ⁰ indicates new water (specular reflection)
        3. Threshold to create binary flood mask

        Args:
            pre_flood: Reference (dry season) SAR backscatter in dB
            during_flood: Flood event SAR backscatter in dB

        Returns:
            Change map (negative values indicate flooding)
        """
        logger.info("Computing SAR change detection (pre-flood vs during-flood)")

        # Log-ratio change detection (already in dB, so subtraction is log-ratio)
        change = during_flood - pre_flood

        change.attrs["description"] = "SAR backscatter change (dB). Negative = potential flooding"
        change.attrs["threshold_note"] = "Δσ⁰ < -3 dB typically indicates flood"

        return change

    @staticmethod
    def _sign_planetary_computer(request):
        """Sign requests for Planetary Computer data access."""
        try:
            import planetary_computer
            return planetary_computer.sign(request)
        except ImportError:
            logger.warning(
                "planetary-computer package not installed. "
                "Using unsigned access (may fail for some assets)."
            )
            return request
