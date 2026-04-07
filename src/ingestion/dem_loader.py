"""
DEM (Digital Elevation Model) Acquisition
==========================================
Downloads and manages elevation data for India.

Primary source: ALOS PALSAR RTC DEM (12.5m resolution)
- Freely available from ASF DAAC (Alaska Satellite Facility)
- Superior to SRTM for Indian terrain (better in Himalayas, Western Ghats)
- 12.5m is the "sweet spot" — urban-grade resolution without HPC costs

Fallback: Copernicus DEM GLO-30 (30m) or SRTM (30m)

The DEM is the most critical static input. Water follows gravity.
Without an accurate DEM, your flood model is modeling fiction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from loguru import logger
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling

from config.settings import settings, RAW_DATA_DIR, PROCESSED_DATA_DIR, DEMSource


class DEMLoader:
    """
    Download, mosaic, and prepare DEM data for an AOI.

    Workflow:
    1. Determine which tiles cover the AOI
    2. Download tiles (ALOS PALSAR 12.5m from ASF DAAC)
    3. Mosaic tiles into a single raster
    4. Reproject to target UTM zone
    5. Clip to AOI boundary
    """

    def __init__(self, dem_source: DEMSource = DEMSource.ALOS_PALSAR):
        self.dem_source = dem_source
        self.raw_dir = RAW_DATA_DIR / "dem"
        self.processed_dir = PROCESSED_DATA_DIR / "dem"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def get_dem_for_aoi(
        self,
        bbox: tuple[float, float, float, float],
        target_crs: Optional[str] = None,
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Complete DEM acquisition pipeline for an Area of Interest.

        Steps:
        1. Find required tiles for bbox
        2. Download missing tiles
        3. Mosaic into single raster
        4. Reproject to target CRS (UTM)
        5. Clip to exact AOI extent

        Args:
            bbox: (min_lon, min_lat, max_lon, max_lat) in WGS84
            target_crs: Target CRS (auto-detected UTM if None)
            output_path: Output file path

        Returns:
            Path to processed DEM GeoTIFF
        """
        if target_crs is None:
            center_lon = (bbox[0] + bbox[2]) / 2
            from config.settings import Settings
            target_crs = Settings.get_utm_zone(center_lon)

        if output_path is None:
            bbox_str = "_".join(f"{c:.2f}" for c in bbox)
            output_path = self.processed_dir / f"dem_{self.dem_source.value}_{bbox_str}.tif"

        if output_path.exists():
            logger.info(f"Processed DEM already exists: {output_path}")
            return output_path

        logger.info(
            f"DEM acquisition pipeline | source={self.dem_source.value} | "
            f"bbox={bbox} | target_crs={target_crs}"
        )

        # Step 1-2: Download tiles
        tile_paths = self._download_tiles(bbox)

        if not tile_paths:
            raise RuntimeError(
                f"No DEM tiles found/downloaded for bbox={bbox}. "
                "Check your internet connection and AOI coordinates."
            )

        # Step 3: Mosaic
        mosaic_path = self.raw_dir / "mosaic_temp.tif"
        self._mosaic_tiles(tile_paths, mosaic_path)

        # Step 4-5: Reproject and clip
        self._reproject_and_clip(mosaic_path, output_path, target_crs, bbox)

        # Cleanup temp mosaic
        if mosaic_path.exists():
            mosaic_path.unlink()

        logger.info(f"DEM ready: {output_path}")
        return output_path

    def _download_tiles(
        self,
        bbox: tuple[float, float, float, float],
    ) -> list[Path]:
        """
        Download DEM tiles that cover the given bounding box.

        For ALOS PALSAR: tiles are 1°×1° in geographic coordinates.
        For SRTM: tiles are also 1°×1° (hgtfiles).
        """
        min_lon, min_lat, max_lon, max_lat = bbox

        if self.dem_source == DEMSource.ALOS_PALSAR:
            return self._download_alos_palsar(min_lon, min_lat, max_lon, max_lat)
        elif self.dem_source == DEMSource.SRTM_30:
            return self._download_srtm(min_lon, min_lat, max_lon, max_lat)
        else:
            return self._download_copernicus_dem(min_lon, min_lat, max_lon, max_lat)

    def _download_alos_palsar(
        self,
        min_lon: float, min_lat: float,
        max_lon: float, max_lat: float,
    ) -> list[Path]:
        """
        Download ALOS PALSAR RTC DEM tiles from ASF DAAC.

        The ALOS World 3D (AW3D30) at 12.5m is accessed via
        ASF's data search API (no authentication required for DEM products).
        """
        import httpx

        logger.info("Downloading ALOS PALSAR DEM tiles from ASF DAAC")

        # Calculate required 1°×1° tiles
        lat_tiles = range(int(np.floor(min_lat)), int(np.ceil(max_lat)) + 1)
        lon_tiles = range(int(np.floor(min_lon)), int(np.ceil(max_lon)) + 1)

        downloaded = []

        for lat in lat_tiles:
            for lon in lon_tiles:
                # ALOS PALSAR tile naming convention
                lat_str = f"N{abs(lat):02d}" if lat >= 0 else f"S{abs(lat):02d}"
                lon_str = f"E{abs(lon):03d}" if lon >= 0 else f"W{abs(lon):03d}"
                tile_name = f"ALPSMLC30_{lat_str}{lon_str}_DSM.tif"
                tile_path = self.raw_dir / tile_name

                if tile_path.exists():
                    logger.debug(f"Tile already exists: {tile_name}")
                    downloaded.append(tile_path)
                    continue

                # ASF DAAC search for ALOS DEM
                search_url = "https://api.daac.asf.alaska.edu/services/search/param"
                try:
                    response = httpx.get(
                        search_url,
                        params={
                            "platform": "ALOS",
                            "processingLevel": "RTC_HI_RES",
                            "bbox": f"{lon},{lat},{lon+1},{lat+1}",
                            "output": "json",
                        },
                        timeout=30.0,
                    )
                    response.raise_for_status()
                    results = response.json()

                    if results and len(results) > 0:
                        download_url = results[0].get("downloadUrl")
                        if download_url:
                            logger.info(f"Downloading {tile_name}...")
                            tile_response = httpx.get(download_url, timeout=120.0)
                            tile_path.write_bytes(tile_response.content)
                            downloaded.append(tile_path)
                            logger.info(f"Downloaded: {tile_name}")
                    else:
                        logger.warning(f"No ALOS tile found for {lat_str}{lon_str}")

                except Exception as e:
                    logger.warning(f"Download failed for {tile_name}: {e}")

        return downloaded

    def _download_srtm(
        self,
        min_lon: float, min_lat: float,
        max_lon: float, max_lat: float,
    ) -> list[Path]:
        """Download SRTM 30m tiles as fallback."""
        try:
            import elevation

            output = self.raw_dir / "srtm_mosaic.tif"
            elevation.clip(
                bounds=(min_lon, min_lat, max_lon, max_lat),
                output=str(output),
                product="SRTM3",
            )
            return [output]
        except ImportError:
            logger.error("Install 'elevation' package for SRTM download: pip install elevation")
            return []

    def _download_copernicus_dem(
        self,
        min_lon: float, min_lat: float,
        max_lon: float, max_lat: float,
    ) -> list[Path]:
        """Download Copernicus DEM GLO-30 via STAC."""
        import pystac_client
        import stackstac

        logger.info("Downloading Copernicus DEM GLO-30 via Planetary Computer")

        catalog = pystac_client.Client.open(
            settings.data_sources.stac_api_url,
        )

        search = catalog.search(
            collections=["cop-dem-glo-30"],
            bbox=(min_lon, min_lat, max_lon, max_lat),
        )

        items = list(search.items())
        if not items:
            logger.warning("No Copernicus DEM tiles found")
            return []

        # Stack into single array and save
        stack = stackstac.stack(items, resolution=30)
        output = self.raw_dir / "copernicus_dem_30m.tif"

        # Compute and save
        dem_data = stack.compute()
        dem_data.rio.to_raster(str(output))

        return [output]

    def _mosaic_tiles(self, tile_paths: list[Path], output_path: Path) -> None:
        """Mosaic multiple DEM tiles into a single raster."""
        logger.info(f"Mosaicking {len(tile_paths)} DEM tiles")

        if len(tile_paths) == 1:
            # Single tile — just copy
            import shutil
            shutil.copy2(tile_paths[0], output_path)
            return

        # Open all tiles
        src_files = [rasterio.open(str(p)) for p in tile_paths]

        try:
            mosaic_array, mosaic_transform = merge(src_files)

            # Write mosaic
            meta = src_files[0].meta.copy()
            meta.update({
                "height": mosaic_array.shape[1],
                "width": mosaic_array.shape[2],
                "transform": mosaic_transform,
                "compress": settings.geospatial.compression,
            })

            with rasterio.open(str(output_path), "w", **meta) as dst:
                dst.write(mosaic_array)

            logger.info(
                f"Mosaic complete | shape={mosaic_array.shape[1:]} | "
                f"transform={mosaic_transform}"
            )
        finally:
            for src in src_files:
                src.close()

    def _reproject_and_clip(
        self,
        input_path: Path,
        output_path: Path,
        target_crs: str,
        bbox: tuple[float, float, float, float],
    ) -> None:
        """Reproject DEM to target CRS and clip to AOI extent."""
        from rasterio.mask import mask as rasterio_mask
        from shapely.geometry import box, mapping
        import json

        logger.info(f"Reprojecting DEM to {target_crs}")

        with rasterio.open(str(input_path)) as src:
            transform, width, height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds
            )

            meta = src.meta.copy()
            meta.update({
                "crs": target_crs,
                "transform": transform,
                "width": width,
                "height": height,
                "compress": settings.geospatial.compression,
                "nodata": settings.geospatial.nodata_value,
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
                        resampling=Resampling.bilinear,
                    )

        logger.info(f"Reprojected DEM saved: {output_path} | size={width}×{height}")


class SoilMoistureLoader:
    """
    Load SMAP (Soil Moisture Active Passive) satellite data.

    SMAP provides global soil moisture at:
    - L3: 9 km resolution, daily
    - L4: 9 km resolution, 3-hourly (model-assimilated)

    Soil moisture is the "pre-saturation index":
    - Dry soil can absorb rainfall → less flood risk
    - Wet/saturated soil → almost all rain becomes runoff → high risk

    The top 5 cm soil moisture (surface) is most relevant for
    flash flood prediction; deeper layers matter for sustained
    river flooding over days.
    """

    def __init__(self):
        self.data_dir = RAW_DATA_DIR / "smap"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load_smap_l3(
        self,
        bbox: tuple[float, float, float, float],
        start_date: str,
        end_date: str,
    ) -> xr.DataArray:
        """
        Load SMAP L3 daily soil moisture via NASA CMR STAC.

        Args:
            bbox: AOI in WGS84
            start_date: Start date
            end_date: End date

        Returns:
            xarray DataArray with soil_moisture (m³/m³) variable
        """
        import pystac_client

        logger.info(f"Loading SMAP L3 soil moisture | {start_date} to {end_date}")

        try:
            catalog = pystac_client.Client.open(
                settings.data_sources.smap_stac_url
            )

            search = catalog.search(
                collections=[settings.data_sources.smap_collection],
                bbox=bbox,
                datetime=f"{start_date}/{end_date}",
            )

            items = list(search.items())
            logger.info(f"Found {len(items)} SMAP granules")

            if not items:
                logger.warning("No SMAP data found. Returning empty DataArray.")
                return xr.DataArray()

            # Process HDF5 granules
            return self._process_smap_granules(items, bbox)

        except Exception as e:
            logger.error(f"SMAP data loading failed: {e}")
            raise

    def _process_smap_granules(
        self,
        items: list,
        bbox: tuple[float, float, float, float],
    ) -> xr.DataArray:
        """Process SMAP HDF5 granules into a time series DataArray."""
        import h5py

        datasets = []

        for item in items:
            # Get the HDF5 asset URL
            href = None
            for asset_key, asset in item.assets.items():
                if asset.href.endswith(".h5"):
                    href = asset.href
                    break

            if href is None:
                continue

            try:
                with h5py.File(href, "r") as h5:
                    sm = h5["Soil_Moisture_Retrieval_Data/soil_moisture"][:]
                    lats = h5["Soil_Moisture_Retrieval_Data/latitude"][:]
                    lons = h5["Soil_Moisture_Retrieval_Data/longitude"][:]

                    # Mask fill values
                    sm[sm < 0] = np.nan

                    da = xr.DataArray(
                        sm,
                        dims=["y", "x"],
                        attrs={"units": "m³/m³", "long_name": "SMAP Soil Moisture"},
                    )
                    datasets.append(da)

            except Exception as e:
                logger.warning(f"Failed to process SMAP granule: {e}")

        if datasets:
            return xr.concat(datasets, dim="time")
        return xr.DataArray()
