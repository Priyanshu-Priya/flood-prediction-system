"""
Vector/Watershed Processing Utilities
=======================================
GeoPandas-based operations for watershed delineation, catchment analysis,
and spatial data management.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
from loguru import logger
from shapely.geometry import box, Point


def load_aoi(
    geojson_path: Optional[Path] = None,
    bbox: Optional[tuple[float, float, float, float]] = None,
    name: str = "AOI",
) -> gpd.GeoDataFrame:
    """
    Load Area of Interest from GeoJSON file or bounding box.

    The configurable AOI system: users define their study area
    via GeoJSON polygon or simple bounding box coordinates.

    Args:
        geojson_path: Path to GeoJSON/Shapefile/GeoPackage
        bbox: (min_lon, min_lat, max_lon, max_lat) in WGS84
        name: AOI identifier name

    Returns:
        GeoDataFrame with the AOI polygon
    """
    if geojson_path:
        logger.info(f"Loading AOI from {geojson_path}")
        gdf = gpd.read_file(str(geojson_path))
    elif bbox:
        logger.info(f"Creating AOI from bbox: {bbox}")
        geometry = box(*bbox)
        gdf = gpd.GeoDataFrame(
            {"name": [name]},
            geometry=[geometry],
            crs="EPSG:4326",
        )
    else:
        raise ValueError("Provide either geojson_path or bbox")

    # Ensure WGS84
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    area_km2 = gdf.to_crs(gdf.estimate_utm_crs()).area.sum() / 1e6
    logger.info(f"AOI loaded: {name} | area={area_km2:.1f} km² | features={len(gdf)}")

    return gdf


def compute_catchment_attributes(
    catchments: gpd.GeoDataFrame,
    dem_stats: dict,
    lulc_stats: Optional[dict] = None,
) -> gpd.GeoDataFrame:
    """
    Compute static catchment attributes used as LSTM static features.

    These attributes characterize each watershed's physical properties
    and are concatenated to the LSTM's dynamic input.

    Attributes computed:
    - Area (km²)
    - Mean elevation (m)
    - Mean slope (degrees)
    - Shape factor (elongation ratio)
    - Stream density (km/km²)
    - Urban fraction (from LULC)
    - Forest fraction (from LULC)
    """
    # Compute area in km²
    utm_crs = catchments.estimate_utm_crs()
    catchments_utm = catchments.to_crs(utm_crs)
    catchments["area_km2"] = catchments_utm.area / 1e6

    # Shape factor: elongation ratio = D_circle / L_max
    catchments["perimeter_km"] = catchments_utm.length / 1e3
    catchments["compactness"] = (
        4 * np.pi * catchments["area_km2"]
        / (catchments["perimeter_km"] ** 2 + 1e-6)
    )

    # Add DEM statistics if provided
    if dem_stats:
        for key, value in dem_stats.items():
            catchments[key] = value

    # Add LULC fractions if provided
    if lulc_stats:
        for key, value in lulc_stats.items():
            catchments[key] = value

    logger.info(f"Catchment attributes computed: {list(catchments.columns)}")
    return catchments


def create_gauge_points(
    stations_df: "pd.DataFrame",
    lat_col: str = "lat",
    lon_col: str = "lon",
) -> gpd.GeoDataFrame:
    """
    Convert station metadata DataFrame to GeoDataFrame with point geometries.

    Used for spatial joining gauges to their parent watersheds.
    """
    geometry = [Point(xy) for xy in zip(stations_df[lon_col], stations_df[lat_col])]
    gdf = gpd.GeoDataFrame(stations_df, geometry=geometry, crs="EPSG:4326")

    logger.info(f"Created {len(gdf)} gauge point geometries")
    return gdf


def spatial_join_gauges_to_catchments(
    gauges: gpd.GeoDataFrame,
    catchments: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Assign each gauge station to its parent watershed via spatial join.

    Returns gauges GeoDataFrame with added catchment attributes.
    """
    joined = gpd.sjoin(gauges, catchments, how="left", predicate="within")
    n_matched = joined.dropna(subset=["index_right"]).shape[0]
    logger.info(f"Spatial join: {n_matched}/{len(gauges)} gauges matched to catchments")
    return joined


def compute_distance_matrix(
    points: gpd.GeoDataFrame,
) -> np.ndarray:
    """
    Compute pairwise distance matrix between point features (in meters).

    Useful for spatial autocorrelation analysis and kriging.
    """
    utm_crs = points.estimate_utm_crs()
    points_utm = points.to_crs(utm_crs)

    coords = np.array([(g.x, g.y) for g in points_utm.geometry])
    n = len(coords)

    dist_matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt((coords[i, 0] - coords[j, 0])**2 + (coords[i, 1] - coords[j, 1])**2)
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    return dist_matrix


def export_flood_zones(
    probability_map: np.ndarray,
    transform,
    crs: str,
    threshold: float = 0.5,
    output_path: Optional[Path] = None,
) -> gpd.GeoDataFrame:
    """
    Vectorize flood probability raster into polygon flood zones.

    Converts raster flood probability map into vector polygons
    for GIS overlay and area calculations.
    """
    from rasterio.features import shapes
    from shapely.geometry import shape

    binary_mask = (probability_map >= threshold).astype(np.uint8)

    polygons = []
    values = []

    for geom, val in shapes(binary_mask, transform=transform):
        if val == 1:
            polygons.append(shape(geom))
            values.append(val)

    if not polygons:
        logger.warning("No flood zones found above threshold")
        return gpd.GeoDataFrame()

    gdf = gpd.GeoDataFrame(
        {"flood_zone": values},
        geometry=polygons,
        crs=crs,
    )

    # Compute area
    utm = gdf.estimate_utm_crs()
    gdf["area_km2"] = gdf.to_crs(utm).area / 1e6

    logger.info(
        f"Vectorized {len(gdf)} flood zones | "
        f"total area={gdf['area_km2'].sum():.2f} km²"
    )

    if output_path:
        gdf.to_file(str(output_path), driver="GeoJSON")
        logger.info(f"Flood zones saved: {output_path}")

    return gdf
