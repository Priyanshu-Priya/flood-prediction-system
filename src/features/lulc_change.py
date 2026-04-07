"""
Land Use / Land Cover Change Detection
========================================
Detects LULC changes that affect flood susceptibility.

Key insight: Urbanization is the #1 human driver of increased flood risk.
- Concrete/asphalt → near-zero infiltration → 95% runoff coefficient
- Natural soil → 20-60% infiltration → 40-80% runoff coefficient
- Deforestation → reduced evapotranspiration → more surface water

Source: ESA WorldCover (10m resolution, global, annual)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
import xarray as xr
from loguru import logger

from config.settings import PROCESSED_DATA_DIR, settings


# ESA WorldCover classification scheme
WORLDCOVER_CLASSES = {
    10: ("Tree Cover", 0.40),           # Runoff coefficient
    20: ("Shrubland", 0.50),
    30: ("Grassland", 0.55),
    40: ("Cropland", 0.60),
    50: ("Built-up", 0.95),             # Nearly impervious
    60: ("Bare / Sparse", 0.70),
    70: ("Snow and Ice", 0.30),
    80: ("Permanent Water", 1.00),
    90: ("Herbaceous Wetland", 0.80),
    95: ("Mangroves", 0.35),
    100: ("Moss and Lichen", 0.50),
}


class LULCChangeDetector:
    """
    Detect land cover changes between two epochs and compute
    hydrologically relevant metrics.

    Key outputs:
    1. LULC classification map (latest)
    2. Impervious surface fraction (per grid cell)
    3. Change detection map (what changed between epochs)
    4. Runoff coefficient map (based on land cover)
    """

    def __init__(self):
        self.output_dir = PROCESSED_DATA_DIR / "lulc"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_worldcover(
        self,
        bbox: tuple[float, float, float, float],
        year: int = 2021,
    ) -> xr.DataArray:
        """
        Load ESA WorldCover via STAC/Planetary Computer.

        Args:
            bbox: AOI in WGS84
            year: WorldCover epoch (2020 or 2021 available)

        Returns:
            xarray DataArray with LULC class values (10m resolution)
        """
        import pystac_client
        import stackstac

        logger.info(f"Loading ESA WorldCover {year} | bbox={bbox}")

        try:
            catalog = pystac_client.Client.open(
                settings.data_sources.stac_api_url,
                modifier=self._sign_planetary_computer,
            )

            search = catalog.search(
                collections=["esa-worldcover"],
                bbox=bbox,
                datetime=f"{year}",
            )

            items = list(search.items())
            if not items:
                raise ValueError(f"No WorldCover data for bbox={bbox}, year={year}")

            logger.info(f"Found {len(items)} WorldCover tiles")

            stack = stackstac.stack(
                items,
                assets=["map"],
                bounds_latlon=bbox,
                resolution=10,
                dtype=np.uint8,
            )

            # Take the first (only) band per pixel
            lulc = stack.isel(time=0).squeeze()
            lulc.attrs["source"] = f"ESA WorldCover {year}"

            return lulc

        except Exception as e:
            logger.error(f"WorldCover load failed: {e}")
            raise

    def compute_impervious_fraction(
        self,
        lulc: xr.DataArray | np.ndarray,
        aggregation_factor: int = 10,
    ) -> np.ndarray:
        """
        Compute fractional impervious surface area.

        From 10m LULC, compute what fraction of each larger grid cell
        is covered by impervious surfaces (built-up class = 50).

        A 10× aggregation gives 100m cells, which is a useful
        resolution for flood modeling.

        Args:
            lulc: Land cover classification (ESA WorldCover codes)
            aggregation_factor: How many LULC cells make one output cell

        Returns:
            Impervious fraction (0-1) at coarser resolution
        """
        if isinstance(lulc, xr.DataArray):
            data = lulc.values
        else:
            data = lulc

        # Binary impervious mask (Built-up = 50)
        impervious = (data == 50).astype(np.float32)

        # Aggregate by taking the mean in each block
        h, w = impervious.shape
        new_h = h // aggregation_factor
        new_w = w // aggregation_factor

        # Trim to exact multiple
        impervious_trimmed = impervious[:new_h * aggregation_factor, :new_w * aggregation_factor]

        # Reshape and compute block means
        fraction = impervious_trimmed.reshape(
            new_h, aggregation_factor, new_w, aggregation_factor
        ).mean(axis=(1, 3))

        logger.info(
            f"Impervious fraction | mean={fraction.mean():.3f} | "
            f"max={fraction.max():.3f} | urban_fraction>0.5: "
            f"{(fraction > 0.5).mean():.3f}"
        )

        return fraction

    def compute_runoff_coefficient_map(
        self,
        lulc: xr.DataArray | np.ndarray,
    ) -> np.ndarray:
        """
        Compute spatially distributed runoff coefficient from LULC.

        The runoff coefficient C determines what fraction of rainfall
        becomes surface runoff: Q_surface = C × P

        This is used in the Rational Method for peak discharge estimation:
            Q_peak = C × i × A
        where i = rainfall intensity, A = catchment area
        """
        if isinstance(lulc, xr.DataArray):
            data = lulc.values
        else:
            data = lulc

        # Create lookup table
        runoff_map = np.full_like(data, 0.5, dtype=np.float32)

        for class_code, (name, coefficient) in WORLDCOVER_CLASSES.items():
            runoff_map[data == class_code] = coefficient

        logger.info(
            f"Runoff coefficient map | mean C={runoff_map.mean():.3f} | "
            f"range=[{runoff_map.min():.2f}, {runoff_map.max():.2f}]"
        )

        return runoff_map

    def detect_changes(
        self,
        lulc_epoch1: xr.DataArray | np.ndarray,
        lulc_epoch2: xr.DataArray | np.ndarray,
    ) -> dict[str, np.ndarray]:
        """
        Detect land cover changes between two epochs.

        Returns change statistics and maps:
        - change_mask: binary (1=changed, 0=unchanged)
        - urbanization_mask: areas that became built-up
        - deforestation_mask: areas that lost tree cover

        Args:
            lulc_epoch1: Earlier LULC classification
            lulc_epoch2: Later LULC classification

        Returns:
            Dictionary of change detection outputs
        """
        if isinstance(lulc_epoch1, xr.DataArray):
            data1 = lulc_epoch1.values
        else:
            data1 = lulc_epoch1

        if isinstance(lulc_epoch2, xr.DataArray):
            data2 = lulc_epoch2.values
        else:
            data2 = lulc_epoch2

        # General change mask
        change_mask = (data1 != data2).astype(np.uint8)

        # Urbanization: anything → built-up (class 50)
        urbanization = ((data1 != 50) & (data2 == 50)).astype(np.uint8)

        # Deforestation: tree cover (10) → anything else
        deforestation = ((data1 == 10) & (data2 != 10)).astype(np.uint8)

        # Compute change in runoff potential
        rc_epoch1 = self.compute_runoff_coefficient_map(data1)
        rc_epoch2 = self.compute_runoff_coefficient_map(data2)
        rc_change = rc_epoch2 - rc_epoch1  # Positive = increased runoff risk

        total_cells = np.prod(data1.shape)
        logger.info(
            f"LULC Change Detection:\n"
            f"  Total changed: {change_mask.sum()} ({change_mask.mean() * 100:.2f}%)\n"
            f"  Urbanized: {urbanization.sum()} cells\n"
            f"  Deforested: {deforestation.sum()} cells\n"
            f"  Mean ΔC (runoff): {rc_change.mean():.4f}"
        )

        return {
            "change_mask": change_mask,
            "urbanization_mask": urbanization,
            "deforestation_mask": deforestation,
            "runoff_coefficient_change": rc_change,
        }

    @staticmethod
    def _sign_planetary_computer(request):
        try:
            import planetary_computer
            return planetary_computer.sign(request)
        except ImportError:
            return request
