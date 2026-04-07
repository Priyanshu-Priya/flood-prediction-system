"""
Terrain Feature Engineering
============================
Computes hydrologically relevant terrain features from DEMs:

1. TWI (Topographic Wetness Index): ln(a / tan(β))
   - High TWI → low slope + large upslope area → water accumulates here
   - This is your #1 static predictor for flood susceptibility

2. Slope: Rate of elevation change (radians)
   - Steep = fast runoff, low infiltration
   - Flat = ponding risk, slow drainage

3. Flow Accumulation: How much upstream area drains through each cell
   - Identifies channels, nallahs, and flood pathways

4. Distance to Channel: How far each cell is from the nearest stream
   - Cells near channels flood first

Uses WhiteboxTools for hydrologically correct terrain analysis.
All operations support Dask for out-of-core processing on large DEMs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from loguru import logger

from config.settings import settings, PROCESSED_DATA_DIR


class TerrainFeatureExtractor:
    """
    Extract terrain-derived features from a DEM using WhiteboxTools.

    Pipeline:
    1. Fill depressions (hydrological conditioning)
    2. Compute slope (radians)
    3. Compute flow direction & accumulation (D-infinity)
    4. Compute TWI
    5. Extract drainage network
    6. Compute distance to nearest channel
    7. Compute aspect (compass direction of steepest descent)

    All intermediate and final products are saved as GeoTIFFs
    with proper metadata for downstream ML feature stacking.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or (PROCESSED_DATA_DIR / "terrain")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize WhiteboxTools
        try:
            from whitebox import WhiteboxTools
            self.wbt = WhiteboxTools()
            self.wbt.set_verbose_mode(False)
            logger.info("WhiteboxTools initialized successfully")
        except ImportError:
            logger.error(
                "WhiteboxTools not installed. Install with: pip install whitebox"
            )
            raise

    def extract_all_features(
        self,
        dem_path: Path,
        stream_threshold_km2: Optional[float] = None,
    ) -> dict[str, Path]:
        """
        Complete terrain feature extraction pipeline.

        Args:
            dem_path: Path to input DEM GeoTIFF (should be in projected CRS)
            stream_threshold_km2: Min contributing area for stream extraction

        Returns:
            Dictionary mapping feature names to output GeoTIFF paths
        """
        if stream_threshold_km2 is None:
            stream_threshold_km2 = settings.geospatial.stream_threshold_km2

        dem_path = Path(dem_path)
        logger.info(f"Starting terrain feature extraction from {dem_path.name}")

        features = {}

        # Step 1: Fill depressions (hydrological conditioning)
        dem_filled = self._fill_depressions(dem_path)
        features["dem_filled"] = dem_filled

        # Step 2: Slope (radians)
        slope = self._compute_slope(dem_filled)
        features["slope"] = slope

        # Step 3: Aspect
        aspect = self._compute_aspect(dem_filled)
        features["aspect"] = aspect

        # Step 4: Flow direction & accumulation
        flow_accum = self._compute_flow_accumulation(dem_filled)
        features["flow_accumulation"] = flow_accum

        # Step 5: TWI
        twi = self._compute_twi(flow_accum, slope)
        features["twi"] = twi

        # Step 6: Drainage network
        streams = self._extract_streams(flow_accum, dem_filled, stream_threshold_km2)
        features["streams"] = streams

        # Step 7: Distance to channel
        distance = self._compute_distance_to_channel(streams)
        features["distance_to_channel"] = distance

        # Step 8: Curvature (plan + profile)
        curvature = self._compute_curvature(dem_filled)
        features["curvature"] = curvature

        logger.info(f"Terrain features extracted: {list(features.keys())}")
        return features

    def _fill_depressions(self, dem_path: Path) -> Path:
        """
        Remove hydrological sinks from the DEM.

        Uses breach-based method (preferred over fill for large DEMs):
        - Breaching carves a channel through barriers
        - Filling raises all cells in a depression
        - Breaching preserves more original topography

        Without this step, flow accumulation will pool in artificial
        sinks and your drainage network will be wrong.
        """
        output = self.output_dir / "dem_filled.tif"

        if output.exists():
            logger.debug("Filled DEM already exists, skipping")
            return output

        logger.info("Filling DEM depressions (breach method)")

        if settings.geospatial.fill_method == "breach":
            self.wbt.breach_depressions_least_cost(
                dem=str(dem_path),
                output=str(output),
                dist=10,  # Max breach distance (cells)
                fill=True,  # Fill remaining after breaching
            )
        else:
            self.wbt.fill_depressions(
                dem=str(dem_path),
                output=str(output),
            )

        logger.info(f"DEM filled: {output}")
        return output

    def _compute_slope(self, dem_path: Path) -> Path:
        """
        Compute slope in radians from DEM.

        Slope = arctan(√(∂z/∂x)² + (∂z/∂y)²)

        Important: Must be in RADIANS for TWI calculation.
        Vertical units must match horizontal units (meters).
        """
        output = self.output_dir / "slope_radians.tif"

        if output.exists():
            logger.debug("Slope already exists, skipping")
            return output

        logger.info("Computing slope (radians)")
        self.wbt.slope(
            dem=str(dem_path),
            output=str(output),
            units="radians",
        )

        logger.info(f"Slope computed: {output}")
        return output

    def _compute_aspect(self, dem_path: Path) -> Path:
        """Compute aspect (direction of steepest descent) in degrees."""
        output = self.output_dir / "aspect_degrees.tif"

        if output.exists():
            return output

        logger.info("Computing aspect")
        self.wbt.aspect(
            dem=str(dem_path),
            output=str(output),
        )

        return output

    def _compute_flow_accumulation(self, dem_path: Path) -> Path:
        """
        Compute flow accumulation using D-infinity method.

        D-infinity distributes flow among multiple downslope cells
        proportionally to steepest gradient — more physically realistic
        than the simpler D8 method (which forces flow into a single cell).

        Output: Specific Contributing Area (SCA) in map units (m²/m)
        This is the 'a' in the TWI equation.
        """
        output = self.output_dir / "flow_accumulation.tif"

        if output.exists():
            logger.debug("Flow accumulation already exists, skipping")
            return output

        logger.info(f"Computing flow accumulation ({settings.geospatial.flow_method.value})")

        if settings.geospatial.flow_method.value == "d_infinity":
            # D-infinity: two steps — flow pointer, then accumulation
            pointer = self.output_dir / "flow_pointer_dinf.tif"
            self.wbt.d_inf_flow_accumulation(
                dem=str(dem_path),
                output=str(output),
                out_type="Specific Contributing Area",
            )
        else:
            # D8: simpler, faster
            self.wbt.d8_flow_accumulation(
                dem=str(dem_path),
                output=str(output),
                out_type="specific contributing area",
            )

        logger.info(f"Flow accumulation computed: {output}")
        return output

    def _compute_twi(self, sca_path: Path, slope_path: Path) -> Path:
        """
        Compute Topographic Wetness Index.

        TWI = ln(a / tan(β))

        where:
            a = specific contributing area (m²/m) — from flow accumulation
            β = local slope (radians)

        Physical interpretation:
            High TWI → large upslope area + low slope → water pools here
            Low TWI → small upslope area + steep slope → water drains quickly

        Typical flood-prone areas have TWI > 12-15.
        """
        output = self.output_dir / "twi.tif"

        if output.exists():
            logger.debug("TWI already exists, skipping")
            return output

        logger.info("Computing Topographic Wetness Index (TWI)")

        self.wbt.wetness_index(
            sca=str(sca_path),
            slope=str(slope_path),
            output=str(output),
        )

        # Log statistics
        with rasterio.open(str(output)) as src:
            twi_data = src.read(1)
            twi_valid = twi_data[twi_data != src.nodata]
            if len(twi_valid) > 0:
                logger.info(
                    f"TWI stats | min={twi_valid.min():.1f} | "
                    f"max={twi_valid.max():.1f} | mean={twi_valid.mean():.1f} | "
                    f"fraction_TWI>12: {(twi_valid > 12).mean():.3f}"
                )

        return output

    def _extract_streams(
        self,
        flow_accum_path: Path,
        dem_path: Path,
        threshold_km2: float,
    ) -> Path:
        """
        Extract stream/drainage network from flow accumulation.

        A cell is classified as "stream" if its contributing area
        exceeds the threshold (e.g., 1 km²).

        For Indian urban areas ("nallahs"), use a lower threshold
        (e.g., 0.25 km²) to capture minor drainage channels.
        """
        output = self.output_dir / "streams.tif"

        if output.exists():
            return output

        logger.info(f"Extracting streams (threshold={threshold_km2} km²)")

        # Convert km² to cell count based on resolution
        cell_area_m2 = settings.geospatial.dem_resolution_m ** 2
        threshold_cells = (threshold_km2 * 1e6) / cell_area_m2

        self.wbt.extract_streams(
            flow_accum=str(flow_accum_path),
            output=str(output),
            threshold=threshold_cells,
        )

        return output

    def _compute_distance_to_channel(self, streams_path: Path) -> Path:
        """
        Compute Euclidean distance from each cell to the nearest stream.

        Cells closer to channels have higher flood risk.
        This captures the concept of "proximity to flood source."
        """
        output = self.output_dir / "distance_to_channel.tif"

        if output.exists():
            return output

        logger.info("Computing distance to nearest channel")

        self.wbt.euclidean_distance(
            i=str(streams_path),
            output=str(output),
        )

        return output

    def _compute_curvature(self, dem_path: Path) -> Path:
        """
        Compute plan curvature (convergence/divergence of flow).

        Positive curvature: converging flow (valleys, gullies) → flood risk
        Negative curvature: diverging flow (ridges, peaks)
        Zero: planar surfaces
        """
        output = self.output_dir / "curvature.tif"

        if output.exists():
            return output

        logger.info("Computing plan curvature")

        self.wbt.plan_curvature(
            dem=str(dem_path),
            output=str(output),
        )

        return output

    def stack_features(
        self,
        feature_paths: dict[str, Path],
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Stack all terrain features into a multi-band GeoTIFF.

        This creates a single file where each band is one feature,
        ready for ML model input.

        Band order: slope, aspect, twi, flow_accumulation,
                    distance_to_channel, curvature, elevation
        """
        if output_path is None:
            output_path = self.output_dir / "terrain_features_stacked.tif"

        feature_order = [
            "slope", "aspect", "twi", "flow_accumulation",
            "distance_to_channel", "curvature", "dem_filled",
        ]

        logger.info(f"Stacking {len(feature_order)} terrain features")

        # Read the first raster to get metadata
        first_key = next(k for k in feature_order if k in feature_paths)
        with rasterio.open(str(feature_paths[first_key])) as src:
            meta = src.meta.copy()
            meta.update({
                "count": len(feature_order),
                "dtype": "float32",
                "compress": settings.geospatial.compression,
                "nodata": settings.geospatial.nodata_value,
            })

        with rasterio.open(str(output_path), "w", **meta) as dst:
            for i, feature_name in enumerate(feature_order, start=1):
                if feature_name not in feature_paths:
                    logger.warning(f"Feature {feature_name} not available, filling with nodata")
                    dst.write(
                        np.full((meta["height"], meta["width"]), settings.geospatial.nodata_value),
                        i,
                    )
                    continue

                with rasterio.open(str(feature_paths[feature_name])) as src:
                    data = src.read(1).astype(np.float32)
                    dst.write(data, i)
                    dst.set_band_description(i, feature_name)

        logger.info(f"Stacked features saved: {output_path}")
        return output_path
