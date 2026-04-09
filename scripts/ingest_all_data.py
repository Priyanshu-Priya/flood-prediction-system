"""
Master Data Ingestion Script
================================
Downloads all required data for a target AOI:
  1. DEM tiles (SRTM 30m / ALOS PALSAR)
  2. GloFAS historical discharge for Indian gauge stations
  3. Sentinel-1 SAR scenes (via Planetary Computer STAC)
  4. ESA WorldCover LULC

Usage:
    python scripts/ingest_all_data.py --aoi chennai
    python scripts/ingest_all_data.py --bbox 80.0 12.8 80.4 13.2 --name custom
    python scripts/ingest_all_data.py --aoi patna --years 3
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import click
import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.logging_config import setup_logging
from config.settings import settings, RAW_DATA_DIR, PROCESSED_DATA_DIR, DEMSource

# ── Preset AOIs ──
AOI_PRESETS = {
    "chennai": {
        "bbox": (80.0, 12.8, 80.4, 13.25),
        "stations": ["GLOFAS_ADYAR"],
        "description": "Greater Chennai — coastal, flash-flood prone",
    },
    "patna": {
        "bbox": (84.5, 25.3, 85.5, 25.9),
        "stations": ["GLOFAS_PATNA"],
        "description": "Patna — Ganga basin, Bihar floods",
    },
    "mumbai": {
        "bbox": (72.7, 18.85, 73.1, 19.3),
        "stations": ["GLOFAS_MUMBAI"],
        "description": "Mumbai Metropolitan — urban flooding",
    },
    "assam": {
        "bbox": (89.5, 25.5, 96.0, 28.0),
        "stations": ["GLOFAS_DIBRUGARH", "GLOFAS_GUWAHATI"],
        "description": "Brahmaputra Basin — large-scale river flooding",
    },
    "delhi": {
        "bbox": (76.8, 28.3, 77.5, 28.9),
        "stations": ["GLOFAS_DELHI"],
        "description": "Delhi-NCR — Yamuna flooding",
    },
    "all_india": {
        "bbox": (68.0, 6.0, 98.0, 36.0),
        "stations": [
            "GLOFAS_FARAKKA", "GLOFAS_VARANASI", "GLOFAS_PATNA",
            "GLOFAS_DIBRUGARH", "GLOFAS_GUWAHATI", "GLOFAS_VIJAYAWADA",
            "GLOFAS_ADYAR", "GLOFAS_SURAT", "GLOFAS_MUMBAI", "GLOFAS_DELHI",
        ],
        "description": "All major Indian gauge stations",
    },
}


def ingest_dem(bbox: tuple, name: str, source: str = "srtm_30") -> Path | None:
    """Download DEM for the AOI."""
    logger.info("=" * 60)
    logger.info("Step 1/4: DEM Acquisition")
    logger.info("=" * 60)

    try:
        from src.ingestion.dem_loader import DEMLoader

        dem_source = DEMSource(source)
        loader = DEMLoader(dem_source=dem_source)
        dem_path = loader.get_dem_for_aoi(bbox)
        logger.info(f"✓ DEM ready: {dem_path}")
        return dem_path
    except Exception as e:
        logger.error(f"✗ DEM download failed: {e}")
        logger.info("  DEM is needed for terrain features. Continuing without it.")
        return None


def ingest_glofas(
    station_ids: list[str],
    years: int = 3,
) -> dict[str, pd.DataFrame]:
    """Download GloFAS historical discharge for specified stations."""
    logger.info("=" * 60)
    logger.info("Step 2/4: GloFAS River Discharge")
    logger.info("=" * 60)

    from src.ingestion.glofas import GloFASClient

    client = GloFASClient()
    end_date = datetime.now() - timedelta(days=60)  # GloFAS has ~2-month latency
    start_date = end_date - timedelta(days=365 * years)

    results = {}
    for sid in station_ids:
        logger.info(f"\n── Station: {sid} ──")
        try:
            df = client.fetch_water_levels(
                station_id=sid,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
            )
            if not df.empty:
                results[sid] = df

                # Save as parquet for fast reloading
                out_dir = PROCESSED_DATA_DIR / "glofas_timeseries"
                out_dir.mkdir(parents=True, exist_ok=True)
                parquet_path = out_dir / f"{sid}_discharge.parquet"
                df.to_parquet(parquet_path)
                logger.info(
                    f"  ✓ {len(df)} daily observations | "
                    f"discharge: {df['discharge_cumecs'].mean():.0f} m³/s (mean) | "
                    f"saved → {parquet_path.name}"
                )
            else:
                logger.warning(f"  ✗ No data returned for {sid}")
        except Exception as e:
            logger.error(f"  ✗ Failed for {sid}: {e}")

    logger.info(f"\nGloFAS Summary: {len(results)}/{len(station_ids)} stations ingested")
    return results


def ingest_sar(
    bbox: tuple,
    start_date: str = "2024-06-01",
    end_date: str = "2024-09-30",
    max_scenes: int = 20,
) -> int:
    """Search and catalog Sentinel-1 SAR scenes for the AOI."""
    logger.info("=" * 60)
    logger.info("Step 3/4: Sentinel-1 SAR Scenes")
    logger.info("=" * 60)

    try:
        from src.ingestion.sentinel_sar import SentinelSARIngestion

        sar = SentinelSARIngestion()
        scenes = sar.search_scenes(
            bbox=bbox,
            start_date=start_date,
            end_date=end_date,
            max_items=max_scenes,
        )

        # Save scene catalog
        catalog_dir = RAW_DATA_DIR / "sar"
        catalog_dir.mkdir(parents=True, exist_ok=True)
        catalog_path = catalog_dir / "scene_catalog.json"
        with open(catalog_path, "w") as f:
            json.dump(scenes, f, indent=2, default=str)

        logger.info(f"  ✓ {len(scenes)} SAR scenes cataloged → {catalog_path.name}")
        return len(scenes)

    except Exception as e:
        logger.error(f"  ✗ SAR search failed: {e}")
        logger.info("  SAR is optional for initial pipeline. Continuing.")
        return 0


def ingest_lulc(bbox: tuple) -> bool:
    """Load ESA WorldCover LULC for the AOI."""
    logger.info("=" * 60)
    logger.info("Step 4/4: ESA WorldCover LULC")
    logger.info("=" * 60)

    try:
        from src.features.lulc_change import LULCChangeDetector

        lulc_detector = LULCChangeDetector()
        lulc = lulc_detector.load_worldcover(bbox=bbox, year=2021)

        # Compute and save runoff coefficient
        runoff_map = lulc_detector.compute_runoff_coefficient_map(lulc)
        impervious = lulc_detector.compute_impervious_fraction(lulc)

        logger.info(f"  ✓ LULC loaded | shape={lulc.shape}")
        return True

    except Exception as e:
        logger.error(f"  ✗ LULC loading failed: {e}")
        logger.info("  LULC is optional for initial pipeline. Continuing.")
        return False


@click.command()
@click.option("--aoi", type=click.Choice(list(AOI_PRESETS.keys())),
              help="Preset AOI name")
@click.option("--bbox", nargs=4, type=float, help="Custom bbox: min_lon min_lat max_lon max_lat")
@click.option("--name", default="custom", help="Name for custom AOI")
@click.option("--years", default=3, help="Years of GloFAS history to download")
@click.option("--dem-source", default="srtm_30",
              type=click.Choice(["alos_palsar", "srtm_30", "cop_dem_30"]))
@click.option("--skip-dem", is_flag=True, help="Skip DEM download")
@click.option("--skip-sar", is_flag=True, help="Skip SAR catalog")
@click.option("--skip-lulc", is_flag=True, help="Skip LULC download")
@click.option("--quick", is_flag=True, help="Quick mode: 1 year, skip SAR/LULC")
def main(aoi, bbox, name, years, dem_source, skip_dem, skip_sar, skip_lulc, quick):
    """Download all data for flood risk prediction."""
    setup_logging(level="INFO")

    if quick:
        years = 1
        skip_sar = True
        skip_lulc = True

    # Resolve AOI
    if aoi:
        preset = AOI_PRESETS[aoi]
        aoi_bbox = preset["bbox"]
        station_ids = preset["stations"]
        aoi_name = aoi
        logger.info(f"AOI: {preset['description']}")
    elif bbox:
        aoi_bbox = tuple(bbox)
        # Use all stations for custom AOI
        station_ids = [
            "GLOFAS_PATNA", "GLOFAS_VARANASI", "GLOFAS_GUWAHATI",
            "GLOFAS_DELHI", "GLOFAS_ADYAR",
        ]
        aoi_name = name
    else:
        click.echo("Error: Provide either --aoi or --bbox")
        raise SystemExit(1)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"FLOOD RISK DATA INGESTION")
    logger.info(f"AOI:      {aoi_name} | bbox={aoi_bbox}")
    logger.info(f"Stations: {station_ids}")
    logger.info(f"History:  {years} years")
    logger.info(f"{'=' * 60}\n")

    start_time = time.time()
    summary = {"aoi": aoi_name, "bbox": aoi_bbox}

    # Step 1: DEM
    if not skip_dem:
        dem_path = ingest_dem(aoi_bbox, aoi_name, dem_source)
        summary["dem"] = str(dem_path) if dem_path else "FAILED"
    else:
        logger.info("Skipping DEM download")

    # Step 2: GloFAS
    glofas_data = ingest_glofas(station_ids, years=years)
    summary["glofas_stations"] = len(glofas_data)
    summary["glofas_total_obs"] = sum(len(df) for df in glofas_data.values())

    # Step 3: SAR
    if not skip_sar:
        n_scenes = ingest_sar(aoi_bbox)
        summary["sar_scenes"] = n_scenes
    else:
        logger.info("Skipping SAR catalog")

    # Step 4: LULC
    if not skip_lulc:
        lulc_ok = ingest_lulc(aoi_bbox)
        summary["lulc"] = "OK" if lulc_ok else "FAILED"
    else:
        logger.info("Skipping LULC download")

    elapsed = time.time() - start_time

    # Save summary
    summary["elapsed_seconds"] = round(elapsed, 1)
    summary_path = PROCESSED_DATA_DIR / f"ingestion_summary_{aoi_name}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"INGESTION COMPLETE | {elapsed:.0f}s elapsed")
    logger.info(f"Summary saved → {summary_path}")
    for k, v in summary.items():
        logger.info(f"  {k}: {v}")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
