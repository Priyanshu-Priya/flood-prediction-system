"""
Automated DEM Download Script
================================
Downloads ALOS PALSAR 12.5m DEM tiles for a given AOI.

Usage:
    python scripts/download_dem.py --bbox 80.0 12.8 80.4 13.2 --name chennai
    python scripts/download_dem.py --geojson path/to/aoi.geojson
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.logging_config import setup_logging
from src.ingestion.dem_loader import DEMLoader
from config.settings import DEMSource


@click.command()
@click.option("--bbox", nargs=4, type=float, help="min_lon min_lat max_lon max_lat")
@click.option("--geojson", type=click.Path(exists=True), help="GeoJSON AOI file")
@click.option("--name", default="aoi", help="Name for output files")
@click.option("--source", default="alos_palsar",
              type=click.Choice(["alos_palsar", "srtm_30", "cop_dem_30"]))
@click.option("--target-crs", default=None, help="Target CRS (auto-detected if None)")
def main(bbox, geojson, name, source, target_crs):
    """Download and prepare DEM for flood modeling."""
    setup_logging(level="INFO")

    if bbox is None and geojson is None:
        click.echo("Error: Provide either --bbox or --geojson")
        raise SystemExit(1)

    if bbox:
        aoi_bbox = tuple(bbox)
    else:
        import geopandas as gpd
        gdf = gpd.read_file(geojson)
        aoi_bbox = tuple(gdf.total_bounds)

    logger.info(f"DEM Download | name={name} | bbox={aoi_bbox} | source={source}")

    dem_source = DEMSource(source)
    loader = DEMLoader(dem_source=dem_source)

    output_path = loader.get_dem_for_aoi(
        bbox=aoi_bbox,
        target_crs=target_crs,
    )

    logger.info(f"✓ DEM ready: {output_path}")


if __name__ == "__main__":
    main()
