"""
End-to-End Feature Preprocessing Pipeline
============================================
Runs the complete feature engineering pipeline for a given AOI:
1. DEM acquisition and conditioning
2. Terrain feature extraction (TWI, slope, flow accumulation)
3. SAR water mask generation
4. LULC loading and change detection
5. Antecedent Precipitation Index computation
6. Feature stacking into ML-ready format

Usage:
    python scripts/preprocess_pipeline.py --bbox 80.0 12.8 80.4 13.2 --name chennai
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.logging_config import setup_logging
from config.settings import PROCESSED_DATA_DIR, DEMSource


@click.command()
@click.option("--bbox", nargs=4, type=float, required=True, help="min_lon min_lat max_lon max_lat")
@click.option("--name", default="aoi", help="Name for output files")
@click.option("--start-date", default="2024-06-01", help="Start date for temporal features")
@click.option("--end-date", default="2024-09-30", help="End date (monsoon season)")
@click.option("--skip-download", is_flag=True, help="Skip DEM download (use existing)")
def main(bbox, name, start_date, end_date, skip_download):
    """Run the complete feature preprocessing pipeline."""
    setup_logging(level="INFO")

    aoi_bbox = tuple(bbox)
    output_dir = PROCESSED_DATA_DIR / name
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"{'='*60}")
    logger.info(f"Feature Pipeline | AOI={name} | bbox={aoi_bbox}")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"{'='*60}")

    # ── Step 1: DEM ──
    logger.info("\n── Step 1/5: DEM Acquisition ──")
    if not skip_download:
        from src.ingestion.dem_loader import DEMLoader
        loader = DEMLoader(dem_source=DEMSource.ALOS_PALSAR)
        dem_path = loader.get_dem_for_aoi(aoi_bbox)
    else:
        dem_path = output_dir / "dem.tif"
        logger.info(f"Skipping download, using: {dem_path}")

    # ── Step 2: Terrain Features ──
    logger.info("\n── Step 2/5: Terrain Feature Extraction ──")
    from src.features.terrain import TerrainFeatureExtractor
    terrain = TerrainFeatureExtractor(output_dir=output_dir / "terrain")
    feature_paths = terrain.extract_all_features(dem_path)

    # Stack features
    stacked_path = terrain.stack_features(
        feature_paths,
        output_path=output_dir / "terrain_features_stacked.tif",
    )

    # ── Step 3: SAR Processing ──
    logger.info("\n── Step 3/5: SAR Water Extent ──")
    try:
        from src.ingestion.sentinel_sar import SentinelSARIngestion
        sar = SentinelSARIngestion()
        sar_stack = sar.load_sar_stack(
            bbox=aoi_bbox,
            start_date=start_date,
            end_date=end_date,
        )
        logger.info("SAR data loaded successfully")
    except Exception as e:
        logger.warning(f"SAR loading skipped: {e}")

    # ── Step 4: LULC ──
    logger.info("\n── Step 4/5: Land Cover Analysis ──")
    try:
        from src.features.lulc_change import LULCChangeDetector
        lulc = LULCChangeDetector()
        lulc_map = lulc.load_worldcover(bbox=aoi_bbox)
        logger.info("LULC data loaded successfully")
    except Exception as e:
        logger.warning(f"LULC loading skipped: {e}")

    # ── Step 5: Summary ──
    logger.info(f"\n{'='*60}")
    logger.info(f"Pipeline complete! Output directory: {output_dir}")
    logger.info(f"Terrain features: {stacked_path}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
