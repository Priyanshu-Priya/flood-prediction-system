"""
DEM Processing Utilities
=========================
Low-level GDAL/Rasterio operations for Digital Elevation Models.
Handles reprojection, resampling, void filling, and hillshade rendering.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from loguru import logger
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject

from config.settings import settings


def read_dem(filepath: Path) -> tuple[np.ndarray, dict]:
    """
    Read DEM GeoTIFF and return array with metadata.

    Returns:
        (data, meta) — elevation array and rasterio metadata dict
    """
    with rasterio.open(str(filepath)) as src:
        data = src.read(1).astype(np.float32)
        meta = {
            "crs": src.crs,
            "transform": src.transform,
            "width": src.width,
            "height": src.height,
            "bounds": src.bounds,
            "resolution": src.res,
            "nodata": src.nodata,
        }

    # Replace nodata with NaN
    if meta["nodata"] is not None:
        data[data == meta["nodata"]] = np.nan

    logger.info(
        f"DEM loaded | shape={data.shape} | CRS={meta['crs']} | "
        f"res={meta['resolution']} | "
        f"elev_range=[{np.nanmin(data):.1f}, {np.nanmax(data):.1f}] m"
    )

    return data, meta


def reproject_raster(
    input_path: Path,
    output_path: Path,
    target_crs: str,
    target_resolution: Optional[float] = None,
    resampling_method: Resampling = Resampling.bilinear,
) -> Path:
    """
    Reproject a raster to a new CRS with optional resampling.

    Args:
        input_path: Source raster path
        output_path: Destination path
        target_crs: Target CRS (e.g., "EPSG:32644")
        target_resolution: Target resolution in target CRS units (meters)
        resampling_method: Resampling algorithm

    Returns:
        Path to reprojected raster
    """
    with rasterio.open(str(input_path)) as src:
        if target_resolution:
            transform, width, height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds,
                resolution=target_resolution,
            )
        else:
            transform, width, height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds,
            )

        meta = src.meta.copy()
        meta.update({
            "crs": target_crs,
            "transform": transform,
            "width": width,
            "height": height,
            "compress": settings.geospatial.compression,
        })

        with rasterio.open(str(output_path), "w", **meta) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=resampling_method,
                )

    logger.info(f"Reprojected to {target_crs}: {output_path}")
    return output_path


def fill_voids(dem: np.ndarray, method: str = "interpolate") -> np.ndarray:
    """
    Fill voids (NaN/nodata) in DEM using interpolation.

    SRTM data has voids in steep terrain and water bodies.
    ALOS PALSAR is generally better but may still have gaps.
    """
    from scipy.interpolate import griddata

    if not np.any(np.isnan(dem)):
        return dem

    n_voids = np.isnan(dem).sum()
    logger.info(f"Filling {n_voids} void pixels in DEM ({method})")

    rows, cols = np.where(~np.isnan(dem))
    values = dem[rows, cols]

    void_rows, void_cols = np.where(np.isnan(dem))

    if len(void_rows) == 0:
        return dem

    filled_values = griddata(
        (rows, cols), values, (void_rows, void_cols),
        method="linear",
    )

    result = dem.copy()
    result[void_rows, void_cols] = filled_values

    # Fill remaining NaN with nearest neighbor
    still_nan = np.isnan(result)
    if np.any(still_nan):
        nn_values = griddata(
            (rows, cols), values,
            (np.where(still_nan)[0], np.where(still_nan)[1]),
            method="nearest",
        )
        result[still_nan] = nn_values

    return result


def compute_hillshade(
    dem: np.ndarray,
    azimuth: float = 315,
    altitude: float = 45,
    cell_size: float = 12.5,
) -> np.ndarray:
    """
    Compute analytical hillshade for DEM visualization.

    Creates a shaded relief effect for map rendering, not used
    as an ML feature but essential for dashboard visualization.
    """
    az_rad = np.radians(azimuth)
    alt_rad = np.radians(altitude)

    # Gradient
    dy, dx = np.gradient(dem, cell_size)

    slope = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect = np.arctan2(-dx, dy)

    hillshade = (
        np.sin(alt_rad) * np.cos(slope)
        + np.cos(alt_rad) * np.sin(slope) * np.cos(az_rad - aspect)
    )

    # Normalize to 0-255
    hillshade = np.clip(hillshade * 255, 0, 255).astype(np.uint8)

    return hillshade


def get_dem_statistics(dem: np.ndarray) -> dict:
    """Compute summary statistics for a DEM array."""
    valid = dem[~np.isnan(dem)]
    return {
        "min_elevation_m": float(np.min(valid)),
        "max_elevation_m": float(np.max(valid)),
        "mean_elevation_m": float(np.mean(valid)),
        "std_elevation_m": float(np.std(valid)),
        "relief_m": float(np.max(valid) - np.min(valid)),
        "n_void_pixels": int(np.isnan(dem).sum()),
        "void_fraction": float(np.isnan(dem).mean()),
    }
