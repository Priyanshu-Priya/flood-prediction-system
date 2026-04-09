"""
End-to-End Feature Preprocessing Pipeline
============================================
Runs the complete feature engineering pipeline for a given AOI:
1. DEM acquisition and conditioning
2. Terrain feature extraction (TWI, slope, flow accumulation)
3. SAR water mask generation → flood labels for XGBoost
4. LULC loading and runoff coefficient mapping
5. Antecedent Precipitation Index from GloFAS discharge
6. Feature stacking into ML-ready format (GeoTIFF + Parquet)

Usage:
    python scripts/preprocess_pipeline.py --bbox 80.0 12.8 80.4 13.2 --name chennai
    python scripts/preprocess_pipeline.py --bbox 80.0 12.8 80.4 13.2 --name chennai --skip-download
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
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
@click.option("--skip-sar", is_flag=True, help="Skip SAR processing")
@click.option("--skip-lulc", is_flag=True, help="Skip LULC loading")
def main(bbox, name, start_date, end_date, skip_download, skip_sar, skip_lulc):
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

    pipeline_results = {"aoi": name, "bbox": list(aoi_bbox), "steps": {}}
    stacked_path = None
    runoff_map = None
    flood_labels = None

    # ── Step 1: DEM ──
    logger.info("\n── Step 1/6: DEM Acquisition ──")
    dem_path = None
    if not skip_download:
        try:
            from src.ingestion.dem_loader import DEMLoader
            loader = DEMLoader(dem_source=DEMSource.ALOS_PALSAR)
            dem_path = loader.get_dem_for_aoi(aoi_bbox)
            pipeline_results["steps"]["dem"] = {"status": "OK", "path": str(dem_path)}
            logger.info(f"✓ DEM ready: {dem_path}")
        except Exception as e:
            logger.warning(f"✗ DEM download failed: {e}")
            pipeline_results["steps"]["dem"] = {"status": "FAILED", "error": str(e)}
    else:
        # Look for existing DEM in common locations
        for candidate in [
            output_dir / "dem.tif",
            PROCESSED_DATA_DIR / "dem" / "dem_mosaic.tif",
            output_dir / "terrain" / "dem_filled.tif",
        ]:
            if candidate.exists():
                dem_path = candidate
                break
        if dem_path:
            logger.info(f"Using existing DEM: {dem_path}")
        else:
            logger.warning("No existing DEM found. Terrain features will be skipped.")

    # ── Step 2: Terrain Features ──
    logger.info("\n── Step 2/6: Terrain Feature Extraction ──")
    if dem_path and dem_path.exists():
        try:
            from src.features.terrain import TerrainFeatureExtractor
            terrain = TerrainFeatureExtractor(output_dir=output_dir / "terrain")
            feature_paths = terrain.extract_all_features(dem_path)
            stacked_path = terrain.stack_features(
                feature_paths,
                output_path=output_dir / "terrain_features_stacked.tif",
            )
            pipeline_results["steps"]["terrain"] = {
                "status": "OK",
                "features": list(feature_paths.keys()),
                "stacked": str(stacked_path),
            }
            logger.info(f"✓ {len(feature_paths)} terrain features stacked → {stacked_path}")
        except Exception as e:
            logger.warning(f"✗ Terrain extraction failed: {e}")
            pipeline_results["steps"]["terrain"] = {"status": "FAILED", "error": str(e)}
    else:
        logger.info("⊘ Skipping terrain (no DEM available)")
        pipeline_results["steps"]["terrain"] = {"status": "SKIPPED"}

    # ── Step 3: SAR → Flood Labels ──
    logger.info("\n── Step 3/6: SAR Water Extent → Flood Labels ──")
    if not skip_sar:
        try:
            from src.ingestion.sentinel_sar import SentinelSARIngestion
            from src.features.sar_processing import SARWaterExtractor

            sar = SentinelSARIngestion()
            sar_stack = sar.load_sar_stack(
                bbox=aoi_bbox,
                start_date=start_date,
                end_date=end_date,
            )
            logger.info(f"SAR stack loaded: {sar_stack.shape}")

            # Extract water masks as flood labels
            extractor = SARWaterExtractor()
            water_masks = []
            for t in range(sar_stack.shape[0]):
                mask = extractor.extract_water_mask(sar_stack[t])
                water_masks.append(mask)

            if water_masks:
                flood_labels = extractor.compute_flood_frequency(water_masks)
                flood_labels_path = output_dir / "flood_labels.npy"
                np.save(flood_labels_path, flood_labels)
                logger.info(
                    f"✓ Flood frequency map generated from {len(water_masks)} scenes | "
                    f"flood_fraction={flood_labels.mean():.4f}"
                )
                pipeline_results["steps"]["sar"] = {
                    "status": "OK",
                    "n_scenes": len(water_masks),
                    "labels_path": str(flood_labels_path),
                }

        except Exception as e:
            logger.warning(f"✗ SAR processing skipped: {e}")
            pipeline_results["steps"]["sar"] = {"status": "FAILED", "error": str(e)}
    else:
        logger.info("⊘ Skipping SAR processing")
        pipeline_results["steps"]["sar"] = {"status": "SKIPPED"}

    # ── Step 4: LULC → Runoff Coefficients ──
    logger.info("\n── Step 4/6: Land Cover → Runoff Coefficients ──")
    if not skip_lulc:
        try:
            from src.features.lulc_change import LULCChangeDetector

            lulc_detector = LULCChangeDetector()
            lulc_map = lulc_detector.load_worldcover(bbox=aoi_bbox)
            runoff_map = lulc_detector.compute_runoff_coefficient_map(lulc_map)
            impervious = lulc_detector.compute_impervious_fraction(lulc_map)

            # Save
            runoff_path = output_dir / "runoff_coefficient.npy"
            np.save(runoff_path, runoff_map)

            pipeline_results["steps"]["lulc"] = {
                "status": "OK",
                "mean_runoff_coeff": float(runoff_map.mean()),
                "impervious_fraction": float(impervious.mean()),
            }
            logger.info(f"✓ LULC + runoff coefficient computed")

        except Exception as e:
            logger.warning(f"✗ LULC loading skipped: {e}")
            pipeline_results["steps"]["lulc"] = {"status": "FAILED", "error": str(e)}
    else:
        logger.info("⊘ Skipping LULC loading")
        pipeline_results["steps"]["lulc"] = {"status": "SKIPPED"}

    # ── Step 5: Precipitation API from GloFAS ──
    logger.info("\n── Step 5/6: Antecedent Precipitation Index ──")
    try:
        from src.features.precipitation import AntecedentPrecipitationIndex

        glofas_dir = PROCESSED_DATA_DIR / "glofas_timeseries"
        if glofas_dir.exists():
            parquet_files = list(glofas_dir.glob("*.parquet"))
            if parquet_files:
                api_calc = AntecedentPrecipitationIndex(decay_factor=0.90)
                api_results = {}

                for pf in parquet_files:
                    sid = pf.stem.replace("_discharge", "")
                    df = pd.read_parquet(pf)
                    discharge = df["discharge_cumecs"].fillna(0).values

                    api_df = api_calc.compute_multi_scale_api(
                        discharge, windows=[3, 7, 14, 30]
                    )
                    api_results[sid] = api_df

                    api_path = output_dir / f"api_{sid}.parquet"
                    api_df.to_parquet(api_path)

                logger.info(f"✓ API computed for {len(api_results)} stations")
                pipeline_results["steps"]["precipitation_api"] = {
                    "status": "OK",
                    "stations": list(api_results.keys()),
                }
            else:
                logger.info("No GloFAS parquet files found — run ingestion first")
                pipeline_results["steps"]["precipitation_api"] = {"status": "NO_DATA"}
        else:
            logger.info("No GloFAS data directory — run ingestion first")
            pipeline_results["steps"]["precipitation_api"] = {"status": "NO_DATA"}

    except Exception as e:
        logger.warning(f"✗ API computation failed: {e}")
        pipeline_results["steps"]["precipitation_api"] = {"status": "FAILED", "error": str(e)}

    # ── Step 6: XGBoost Feature Export ──
    logger.info("\n── Step 6/6: XGBoost Feature Export ──")
    if stacked_path and stacked_path.exists():
        try:
            import rasterio

            with rasterio.open(str(stacked_path)) as src:
                terrain_data = src.read()  # (bands, H, W)
                n_bands = src.count

            # Flatten spatial → tabular
            h, w = terrain_data.shape[1], terrain_data.shape[2]
            feature_names = [
                "slope", "aspect", "twi", "flow_accumulation",
                "distance_to_channel", "curvature", "elevation",
            ][:n_bands]

            features_flat = {}
            for i, fname in enumerate(feature_names):
                features_flat[fname] = terrain_data[i].ravel()

            # Add runoff coefficient if available
            if runoff_map is not None and runoff_map.size > 0:
                # Resample to match terrain grid if needed
                from scipy.ndimage import zoom
                if runoff_map.shape != (h, w):
                    scale_h = h / runoff_map.shape[0]
                    scale_w = w / runoff_map.shape[1]
                    runoff_resampled = zoom(runoff_map, (scale_h, scale_w), order=0)
                else:
                    runoff_resampled = runoff_map
                features_flat["runoff_coefficient"] = runoff_resampled.ravel()

            X = pd.DataFrame(features_flat)

            # Generate labels
            if flood_labels is not None and flood_labels.size > 0:
                from scipy.ndimage import zoom
                if flood_labels.shape != (h, w):
                    scale_h = h / flood_labels.shape[0]
                    scale_w = w / flood_labels.shape[1]
                    labels_resampled = zoom(flood_labels, (scale_h, scale_w), order=0)
                else:
                    labels_resampled = flood_labels
                y = (labels_resampled.ravel() > 0.1).astype(int)
            else:
                # Generate synthetic labels from TWI as proxy
                # (TWI > 12 is a reasonable proxy for flood-prone areas)
                twi_idx = feature_names.index("twi") if "twi" in feature_names else 2
                twi_flat = terrain_data[twi_idx].ravel()
                y = (twi_flat > np.nanpercentile(twi_flat[np.isfinite(twi_flat)], 80)).astype(int)
                logger.info("  Using TWI-derived proxy labels (no SAR flood masks available)")

            # Remove invalid pixels
            valid = ~X.isna().any(axis=1)
            X_clean = X[valid].reset_index(drop=True)
            y_clean = y[valid]

            # save
            xgb_path = output_dir / "xgboost_features.parquet"
            X_clean.to_parquet(xgb_path)
            np.save(output_dir / "xgboost_labels.npy", y_clean)

            pipeline_results["steps"]["xgboost_export"] = {
                "status": "OK",
                "n_samples": int(len(y_clean)),
                "n_features": int(X_clean.shape[1]),
                "flood_rate": float(y_clean.mean()),
            }
            logger.info(
                f"✓ XGBoost features exported | "
                f"{len(y_clean)} samples × {X_clean.shape[1]} features | "
                f"flood_rate={y_clean.mean():.4f}"
            )

        except Exception as e:
            logger.warning(f"✗ XGBoost export failed: {e}")
            pipeline_results["steps"]["xgboost_export"] = {"status": "FAILED", "error": str(e)}
    else:
        logger.info("⊘ Skipping XGBoost export (no terrain stack)")
        pipeline_results["steps"]["xgboost_export"] = {"status": "SKIPPED"}

    # ── Save pipeline results ──
    results_path = output_dir / "pipeline_results.json"
    with open(results_path, "w") as f:
        json.dump(pipeline_results, f, indent=2, default=str)

    # ── Summary ──
    ok_count = sum(1 for s in pipeline_results["steps"].values() if s.get("status") == "OK")
    total = len(pipeline_results["steps"])

    logger.info(f"\n{'='*60}")
    logger.info(f"PIPELINE COMPLETE | {ok_count}/{total} steps succeeded")
    logger.info(f"Output directory: {output_dir}")
    for step, result in pipeline_results["steps"].items():
        status = result.get("status", "UNKNOWN")
        icon = "✓" if status == "OK" else "✗" if status == "FAILED" else "⊘"
        logger.info(f"  {icon} {step}: {status}")
    logger.info(f"Results: {results_path}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
