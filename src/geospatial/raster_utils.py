"""
Raster I/O Utilities
=====================
High-level raster operations built on Rasterio:
- Tiling large rasters for chunked processing
- Zonal statistics within vector polygons
- GeoTIFF export with compression and overviews
- Coordinate transformations
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from loguru import logger
from rasterio.windows import Window
from rasterio.transform import from_bounds

from config.settings import settings


def tile_raster(
    input_path: Path,
    tile_size: int = 512,
    overlap: int = 0,
) -> list[dict]:
    """
    Split a large raster into processing tiles (windows).

    Returns tile metadata without loading pixels — tiles can be
    processed individually or in parallel with Dask.

    Args:
        input_path: Path to input raster
        tile_size: Tile dimension in pixels
        overlap: Pixel overlap between tiles (for edge effects)

    Returns:
        List of tile metadata dicts with window and bounds info
    """
    with rasterio.open(str(input_path)) as src:
        height, width = src.height, src.width

    tiles = []
    step = tile_size - overlap

    for row in range(0, height, step):
        for col in range(0, width, step):
            win_height = min(tile_size, height - row)
            win_width = min(tile_size, width - col)

            tiles.append({
                "window": Window(col, row, win_width, win_height),
                "row_off": row,
                "col_off": col,
                "height": win_height,
                "width": win_width,
            })

    logger.info(
        f"Tiled {input_path.name} | {width}×{height} → "
        f"{len(tiles)} tiles of {tile_size}×{tile_size}"
    )

    return tiles


def read_tile(
    input_path: Path,
    window: Window,
    band: int = 1,
) -> np.ndarray:
    """Read a specific tile (window) from a raster."""
    with rasterio.open(str(input_path)) as src:
        return src.read(band, window=window).astype(np.float32)


def zonal_statistics(
    raster_path: Path,
    zones: "geopandas.GeoDataFrame",
    stats: list[str] | None = None,
    band: int = 1,
) -> "geopandas.GeoDataFrame":
    """
    Compute zonal statistics of a raster within vector polygon zones.

    Useful for: aggregating TWI, slope, etc. per watershed or admin boundary.

    Args:
        raster_path: Input raster path
        zones: GeoDataFrame with polygon geometries
        stats: List of statistics ("mean", "min", "max", "std", "sum", "count")
        band: Raster band to analyze

    Returns:
        GeoDataFrame with added statistic columns
    """
    import geopandas as gpd
    from rasterio.mask import mask as rasterio_mask
    from shapely.geometry import mapping

    if stats is None:
        stats = ["mean", "min", "max", "std"]

    logger.info(f"Computing zonal stats | {len(zones)} zones | stats={stats}")

    result_cols = {f"zonal_{s}": [] for s in stats}

    with rasterio.open(str(raster_path)) as src:
        for _, zone in zones.iterrows():
            try:
                masked_data, _ = rasterio_mask(
                    src, [mapping(zone.geometry)],
                    crop=True, nodata=np.nan,
                )
                data = masked_data[0]
                valid = data[~np.isnan(data)]

                if len(valid) == 0:
                    for s in stats:
                        result_cols[f"zonal_{s}"].append(np.nan)
                    continue

                stat_funcs = {
                    "mean": np.mean, "min": np.min, "max": np.max,
                    "std": np.std, "sum": np.sum, "count": len,
                }

                for s in stats:
                    result_cols[f"zonal_{s}"].append(float(stat_funcs[s](valid)))

            except Exception:
                for s in stats:
                    result_cols[f"zonal_{s}"].append(np.nan)

    for col_name, values in result_cols.items():
        zones[col_name] = values

    return zones


def save_geotiff(
    data: np.ndarray,
    output_path: Path,
    crs: str,
    transform,
    nodata: float = -9999.0,
    band_names: Optional[list[str]] = None,
    compress: str = "lzw",
) -> Path:
    """
    Save numpy array as a GeoTIFF with proper metadata.

    Args:
        data: 2D (H, W) or 3D (bands, H, W) array
        output_path: Output file path
        crs: Coordinate reference system string
        transform: Rasterio affine transform
        nodata: NoData value
        band_names: Optional band description names
        compress: Compression method

    Returns:
        Path to saved file
    """
    if data.ndim == 2:
        data = data[np.newaxis, :, :]

    n_bands, height, width = data.shape

    meta = {
        "driver": "GTiff",
        "dtype": data.dtype,
        "width": width,
        "height": height,
        "count": n_bands,
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
        "compress": compress,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }

    with rasterio.open(str(output_path), "w", **meta) as dst:
        for i in range(n_bands):
            dst.write(data[i], i + 1)
            if band_names and i < len(band_names):
                dst.set_band_description(i + 1, band_names[i])

    logger.info(f"Saved GeoTIFF: {output_path} | {n_bands} bands × {width}×{height}")
    return output_path


def pixel_to_coords(
    row: int, col: int, transform
) -> tuple[float, float]:
    """Convert pixel row/col to geographic coordinates."""
    x, y = rasterio.transform.xy(transform, row, col)
    return x, y


def coords_to_pixel(
    x: float, y: float, transform
) -> tuple[int, int]:
    """Convert geographic coordinates to pixel row/col."""
    row, col = rasterio.transform.rowcol(transform, x, y)
    return row, col
