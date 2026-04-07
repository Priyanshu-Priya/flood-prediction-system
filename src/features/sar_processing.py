"""
SAR Backscatter Processing → Flood Water Extent Masks
======================================================
Converts Sentinel-1 SAR imagery into binary flood maps.

Physics of SAR flood detection:
1. Radar signal hits smooth water surface → specular reflection
   → almost no signal returns to satellite → LOW backscatter (dark)
2. Radar hits rough surface (soil, vegetation, buildings)
   → diffuse scattering → HIGH backscatter (bright)
3. Threshold: σ⁰_VV < -16 dB typically indicates open water

Processing chain:
1. Speckle filtering (SAR images are inherently noisy)
2. Histogram-based water thresholding (Otsu's method)
3. Morphological cleanup (remove salt-and-pepper noise)
4. Vectorization (raster → flood extent polygons)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
import xarray as xr
from loguru import logger
from scipy import ndimage
from scipy.ndimage import binary_opening, binary_closing

from config.settings import settings, PROCESSED_DATA_DIR


class SARWaterExtractor:
    """
    Extract flood water extent from Sentinel-1 SAR backscatter imagery.

    The process assumes input is radiometrically terrain-corrected (RTC)
    SAR data with values in decibels (dB). When using STAC-sourced
    Sentinel-1 RTC products, this is already handled.
    """

    def __init__(self):
        self.output_dir = PROCESSED_DATA_DIR / "sar"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_water_mask(
        self,
        sar_vv: np.ndarray | xr.DataArray,
        method: str = "otsu",
        manual_threshold_db: Optional[float] = None,
    ) -> np.ndarray:
        """
        Generate binary water mask from SAR VV backscatter.

        Water Mask = 1 where σ⁰_VV < threshold, else 0

        Args:
            sar_vv: SAR backscatter in VV polarization (dB)
            method: Thresholding method ("otsu", "manual", "adaptive")
            manual_threshold_db: Fixed threshold if method="manual"

        Returns:
            Binary water mask (1=water, 0=non-water)
        """
        if isinstance(sar_vv, xr.DataArray):
            data = sar_vv.values.copy()
        else:
            data = sar_vv.copy()

        # Handle NaN and convert to dB if needed (check if values are in linear scale)
        valid_mask = ~np.isnan(data) & (data != 0)

        if np.nanmax(data) > 1 and np.nanmin(data[valid_mask]) > 0:
            # Data might be in linear scale, convert to dB
            data[valid_mask] = 10 * np.log10(data[valid_mask])
            logger.debug("Converted SAR from linear to dB scale")

        # Step 1: Speckle filtering
        filtered = self._speckle_filter(data)

        # Step 2: Determine threshold
        if method == "otsu":
            threshold = self._otsu_threshold(filtered[valid_mask])
            logger.info(f"Otsu threshold: {threshold:.1f} dB")
        elif method == "manual":
            threshold = manual_threshold_db or settings.geospatial.sar_water_threshold_db
            logger.info(f"Manual threshold: {threshold:.1f} dB")
        elif method == "adaptive":
            threshold = self._adaptive_threshold(filtered)
            logger.info(f"Adaptive threshold computed")
        else:
            raise ValueError(f"Unknown thresholding method: {method}")

        # Step 3: Apply threshold
        water_mask = np.zeros_like(data, dtype=np.uint8)
        water_mask[filtered < threshold] = 1
        water_mask[~valid_mask] = 0  # No data areas are not water

        # Step 4: Morphological cleanup
        water_mask = self._morphological_cleanup(water_mask)

        water_fraction = water_mask.sum() / valid_mask.sum() * 100
        logger.info(f"Water extent: {water_fraction:.2f}% of valid area")

        return water_mask

    def _speckle_filter(
        self,
        sar_db: np.ndarray,
        filter_type: Optional[str] = None,
        window_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Apply speckle filter to reduce SAR noise.

        SAR images have multiplicative speckle noise due to coherent
        imaging. We need to reduce it without losing edge information.

        Lee Enhanced filter preserves edges better than simple averaging.
        """
        if filter_type is None:
            filter_type = settings.geospatial.sar_speckle_filter
        if window_size is None:
            window_size = settings.geospatial.sar_speckle_window

        logger.debug(f"Applying speckle filter: {filter_type} ({window_size}×{window_size})")

        if filter_type == "lee_enhanced":
            return self._lee_filter(sar_db, window_size)
        elif filter_type == "median":
            return ndimage.median_filter(sar_db, size=window_size)
        elif filter_type == "mean":
            kernel = np.ones((window_size, window_size)) / (window_size ** 2)
            return ndimage.convolve(sar_db, kernel)
        else:
            # Fallback to simple median
            return ndimage.median_filter(sar_db, size=window_size)

    def _lee_filter(self, img: np.ndarray, window_size: int = 5) -> np.ndarray:
        """
        Lee filter for SAR speckle reduction.

        The Lee filter adapts to local statistics:
        - In homogeneous areas: acts like a mean filter (smooths)
        - Near edges: preserves the original value

        filtered = mean + k × (original - mean)
        where k = variance / (variance + noise_variance)
        """
        # Compute local mean
        kernel = np.ones((window_size, window_size)) / (window_size ** 2)
        local_mean = ndimage.convolve(img, kernel)

        # Compute local variance
        local_sq_mean = ndimage.convolve(img ** 2, kernel)
        local_var = local_sq_mean - local_mean ** 2
        local_var = np.maximum(local_var, 0)  # Numerical safety

        # Estimate noise variance (using coefficient of variation)
        # For SAR, ENL (Equivalent Number of Looks) determines noise level
        overall_var = np.nanvar(img)
        noise_var = overall_var * 0.25  # Approximate for 4-look data

        # Compute weighting factor
        k = local_var / (local_var + noise_var + 1e-10)
        k = np.clip(k, 0, 1)

        filtered = local_mean + k * (img - local_mean)

        return filtered

    def _otsu_threshold(self, data: np.ndarray) -> float:
        """
        Compute Otsu's threshold on SAR backscatter histogram.

        Otsu's method finds the threshold that minimizes intra-class
        variance — automatically separating water (low σ⁰) from
        non-water (high σ⁰) in the bimodal SAR histogram.
        """
        # Remove NaN and extreme values
        valid = data[np.isfinite(data)]
        valid = valid[(valid > -40) & (valid < 5)]  # Reasonable dB range

        if len(valid) < 100:
            logger.warning("Insufficient valid pixels for Otsu. Using default threshold.")
            return settings.geospatial.sar_water_threshold_db

        # Histogram
        n_bins = 256
        hist, bin_edges = np.histogram(valid, bins=n_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Otsu's method — maximize inter-class variance
        total = hist.sum()
        sum_total = (hist * bin_centers).sum()

        best_threshold = bin_centers[0]
        best_variance = 0

        sum_bg = 0
        weight_bg = 0

        for i in range(n_bins):
            weight_bg += hist[i]
            if weight_bg == 0:
                continue

            weight_fg = total - weight_bg
            if weight_fg == 0:
                break

            sum_bg += hist[i] * bin_centers[i]
            mean_bg = sum_bg / weight_bg
            mean_fg = (sum_total - sum_bg) / weight_fg

            inter_class_var = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2

            if inter_class_var > best_variance:
                best_variance = inter_class_var
                best_threshold = bin_centers[i]

        return best_threshold

    def _adaptive_threshold(self, sar_db: np.ndarray, block_size: int = 128) -> np.ndarray:
        """
        Compute spatially varying threshold using local Otsu's method.

        Different regions may have different backscatter characteristics
        (urban vs. agricultural vs. forested). Adaptive thresholding
        handles this diversity.
        """
        h, w = sar_db.shape
        threshold_map = np.full_like(sar_db, settings.geospatial.sar_water_threshold_db)

        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                block = sar_db[y:y + block_size, x:x + block_size]
                valid = block[np.isfinite(block)]
                if len(valid) > 50:
                    threshold_map[y:y + block_size, x:x + block_size] = self._otsu_threshold(valid)

        return threshold_map

    def _morphological_cleanup(
        self,
        mask: np.ndarray,
        min_size_pixels: int = 50,
    ) -> np.ndarray:
        """
        Remove noise from water mask using morphological operations.

        1. Opening (erosion → dilation): removes small false positives
        2. Closing (dilation → erosion): fills small holes in water bodies
        3. Size filtering: remove connected components smaller than threshold
        """
        struct = np.ones((3, 3), dtype=np.uint8)

        # Opening removes small bright noise (false water detections)
        cleaned = binary_opening(mask, structure=struct, iterations=1).astype(np.uint8)

        # Closing fills small gaps within water bodies
        cleaned = binary_closing(cleaned, structure=struct, iterations=1).astype(np.uint8)

        # Remove small connected components (< min_size_pixels)
        labeled, n_features = ndimage.label(cleaned)
        for i in range(1, n_features + 1):
            component_size = (labeled == i).sum()
            if component_size < min_size_pixels:
                cleaned[labeled == i] = 0

        removed = mask.sum() - cleaned.sum()
        logger.debug(f"Morphological cleanup: removed {removed} noise pixels")

        return cleaned

    def compute_flood_frequency(
        self,
        water_masks: list[np.ndarray],
    ) -> np.ndarray:
        """
        Compute flood frequency map from a time series of water masks.

        flood_frequency(x,y) = N_flooded(x,y) / N_total

        This gives the fraction of observations where each pixel was wet.
        Areas with high flood frequency are permanently/seasonally wet.
        Areas with occasional flooding are the ones we want to predict.
        """
        if not water_masks:
            raise ValueError("No water masks provided")

        stack = np.stack(water_masks, axis=0).astype(np.float32)
        freq = stack.mean(axis=0)

        logger.info(
            f"Flood frequency map | {len(water_masks)} observations | "
            f"max_freq={freq.max():.3f} | mean_freq={freq.mean():.4f}"
        )

        return freq

    def save_water_mask(
        self,
        mask: np.ndarray,
        reference_raster: Path | str,
        output_path: Path | str,
    ) -> Path:
        """Save water mask as GeoTIFF using reference raster for spatial metadata."""
        output_path = Path(output_path)

        with rasterio.open(str(reference_raster)) as ref:
            meta = ref.meta.copy()

        meta.update({
            "dtype": "uint8",
            "count": 1,
            "nodata": 255,
            "compress": "lzw",
        })

        with rasterio.open(str(output_path), "w", **meta) as dst:
            dst.write(mask.astype(np.uint8), 1)
            dst.set_band_description(1, "water_mask (1=water, 0=non-water)")

        logger.info(f"Water mask saved: {output_path}")
        return output_path
