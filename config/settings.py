"""
Flood Risk Prediction System — Central Configuration
=====================================================
India-centric settings for data sources, model hyperparameters,
geospatial processing, and deployment configuration.

All paths, API endpoints, and tunable parameters are centralized here.
Uses pydantic-settings for environment variable overrides via .env file.
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


# ──────────────────────────────────────────────────────────────
# Enumerations
# ──────────────────────────────────────────────────────────────

class FloodType(str, Enum):
    FLUVIAL = "fluvial"    # River flooding
    PLUVIAL = "pluvial"    # Flash / urban flooding
    COASTAL = "coastal"    # Storm surge (future extension)


class DEMSource(str, Enum):
    ALOS_PALSAR = "alos_palsar"   # 12.5m — recommended for India
    COPERNICUS_30 = "cop_dem_30"  # 30m — global fallback
    SRTM_30 = "srtm_30"          # 30m — legacy


class FlowMethod(str, Enum):
    D8 = "d8"
    D_INF = "d_infinity"
    RHO8 = "rho8"


# ──────────────────────────────────────────────────────────────
# Project Paths
# ──────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
for _dir in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, OUTPUT_DIR, LOG_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────
# India-Centric Data Sources
# ──────────────────────────────────────────────────────────────

class DataSourceSettings(BaseSettings):
    """API endpoints and credentials for data sources."""

    # NASA GPM (Global Precipitation Measurement) — 30-min rainfall
    gpm_base_url: str = "https://gpm.nasa.gov/data"
    gpm_opendap_url: str = "https://gpm1.gesdisc.eosdis.nasa.gov/opendap"
    nasa_earthdata_token: Optional[str] = None

    # SMAP (Soil Moisture Active Passive)
    smap_stac_url: str = "https://cmr.earthdata.nasa.gov/stac/NSIDC_ECS"
    smap_collection: str = "SPL3SMP_E"  # Enhanced L3 daily, 9km resolution

    # ALOS PALSAR DEM (12.5m)
    alos_palsar_url: str = "https://search.asf.alaska.edu"  # ASF DAAC
    alos_dem_resolution: float = 12.5   # meters/pixel

    # Sentinel-1 SAR (via STAC — no GEE needed)
    stac_api_url: str = "https://planetarycomputer.microsoft.com/api/stac/v1"
    sentinel1_collection: str = "sentinel-1-rtc"
    sentinel2_collection: str = "sentinel-2-l2a"

    # Copernicus Data Space (alternative Sentinel source)
    copernicus_dataspace_url: str = "https://dataspace.copernicus.eu"
    copernicus_client_id: Optional[str] = None
    copernicus_client_secret: Optional[str] = None

    # CDSE Specific (new architecture)
    copernicus_execution_url: str = "https://ewds.climate.copernicus.eu/api/retrieve/v1/processes/cems-glofas-historical/execution"

    # Google Earth Engine (Community Edition)
    gee_project_id: Optional[str] = None
    gee_service_account_json_path: Optional[str] = None

    # GloFAS / Copernicus EWDS (global river discharge)
    glofas_ewds_key: Optional[str] = None
    glofas_dataset: str = "cems-glofas-historical"

    model_config = {"env_prefix": "FLOOD_DATA_", "env_file": ".env", "extra": "ignore"}


# ──────────────────────────────────────────────────────────────
# Geospatial Processing Settings
# ──────────────────────────────────────────────────────────────

class GeospatialSettings(BaseSettings):
    """DEM processing and terrain analysis configuration."""

    dem_source: DEMSource = DEMSource.ALOS_PALSAR
    dem_resolution_m: float = 12.5
    target_crs: str = "EPSG:32644"  # UTM Zone 44N (covers most of India)

    # WhiteboxTools terrain analysis
    flow_method: FlowMethod = FlowMethod.D_INF
    stream_threshold_km2: float = 1.0  # Min contributing area for stream extraction
    fill_method: str = "breach"         # "breach" or "fill" for sink removal

    # Raster processing
    tile_size: int = 512               # Processing tile size (pixels)
    nodata_value: float = -9999.0
    compression: str = "lzw"           # GeoTIFF compression
    chunk_size: int = 1024             # Dask chunk size for lazy loading

    # SAR processing
    sar_speckle_filter: str = "lee_enhanced"
    sar_speckle_window: int = 5         # Kernel size
    sar_water_threshold_db: float = -16.0  # σ⁰_VV threshold for water
    sar_polarization: str = "VV"

    model_config = {"env_prefix": "FLOOD_GEO_", "env_file": ".env", "extra": "ignore"}


# ──────────────────────────────────────────────────────────────
# LSTM Model Hyperparameters
# ──────────────────────────────────────────────────────────────

class LSTMSettings(BaseSettings):
    """Hindcast-Forecast LSTM architecture configuration."""

    # Architecture
    input_features: int = 12           # Number of dynamic input features per timestep
    static_features: int = 8           # Number of static catchment attributes
    d_model: int = 128                 # Hidden dimension
    n_layers: int = 2                  # LSTM layers (per hindcast & forecast modules)
    n_attention_heads: int = 4         # Temporal attention heads
    dropout: float = 0.2

    # Sequence lengths
    lookback_steps: int = 168          # 7 days at hourly resolution
    forecast_steps: int = 72           # 3-day forecast horizon (hourly)
    lead_times: list[int] = Field(default=[6, 12, 24, 48, 72])  # Hours

    # Training
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 200
    patience: int = 20                 # Early stopping patience
    gradient_clip: float = 1.0
    scheduler: str = "cosine"          # "cosine" or "reduce_on_plateau"

    # Device
    device: str = "cuda"               # RTX 4050
    mixed_precision: bool = True       # FP16 training for speed
    num_workers: int = 4               # DataLoader workers

    model_config = {"env_prefix": "FLOOD_LSTM_", "env_file": ".env", "extra": "ignore"}

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        try:
            import torch
            if v == "cuda" and not torch.cuda.is_available():
                return "cpu"
        except ImportError:
            return "cpu"
        return v


# ──────────────────────────────────────────────────────────────
# XGBoost Spatial Model Settings
# ──────────────────────────────────────────────────────────────

class XGBoostSettings(BaseSettings):
    """XGBoost flood susceptibility model configuration."""

    # Hyperparameters (tuned via Optuna)
    n_estimators: int = 1000
    max_depth: int = 8
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 5
    gamma: float = 0.1
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0

    # Training
    early_stopping_rounds: int = 50
    eval_metric: str = "auc"
    tree_method: str = "hist"          # GPU-accelerated on RTX 4050 (via device config)
    device: str = "cuda:0"

    # Spatial validation
    n_spatial_folds: int = 5           # Leave-one-watershed-out CV
    positive_weight_ratio: float = 5.0  # Handle class imbalance (floods are rare)

    model_config = {"env_prefix": "FLOOD_XGB_", "env_file": ".env", "extra": "ignore"}


# ──────────────────────────────────────────────────────────────
# Ensemble Settings
# ──────────────────────────────────────────────────────────────

class EnsembleSettings(BaseSettings):
    """Weighted ensemble combiner configuration."""

    alpha_temporal: float = 0.6        # Weight for LSTM temporal prediction
    alpha_spatial: float = 0.4         # Weight for XGBoost spatial prediction
    calibrate_per_basin: bool = True   # Learn optimal α per watershed
    alert_thresholds: dict = Field(default={
        "green": 0.0,
        "yellow": 0.3,
        "orange": 0.6,
        "red": 0.8,
    })

    model_config = {"env_prefix": "FLOOD_ENS_", "env_file": ".env", "extra": "ignore"}


# ──────────────────────────────────────────────────────────────
# API & Dashboard Settings
# ──────────────────────────────────────────────────────────────

class APISettings(BaseSettings):
    """FastAPI service configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    cors_origins: list[str] = Field(default=["http://localhost:8501"])  # Streamlit
    rate_limit: str = "100/minute"
    api_key: Optional[str] = None
    model_cache_ttl: int = 3600        # Seconds to cache loaded models

    model_config = {"env_prefix": "FLOOD_API_", "env_file": ".env", "extra": "ignore"}


class DashboardSettings(BaseSettings):
    """Streamlit dashboard configuration."""

    api_base_url: str = "http://localhost:8000"
    map_center_lat: float = 20.5937    # Center of India
    map_center_lon: float = 78.9629
    map_default_zoom: int = 5
    refresh_interval_sec: int = 300    # Auto-refresh every 5 minutes

    model_config = {"env_prefix": "FLOOD_DASH_", "env_file": ".env", "extra": "ignore"}


# ──────────────────────────────────────────────────────────────
# Master Configuration Singleton
# ──────────────────────────────────────────────────────────────

class Settings:
    """Aggregated settings — single access point for all configuration."""

    def __init__(self):
        self.data_sources = DataSourceSettings()
        self.geospatial = GeospatialSettings()
        self.lstm = LSTMSettings()
        self.xgboost = XGBoostSettings()
        self.ensemble = EnsembleSettings()
        self.api = APISettings()
        self.dashboard = DashboardSettings()



    # India-specific UTM zone lookup based on longitude
    UTM_ZONES_INDIA: dict[str, str] = {
        "42N": "EPSG:32642",  # Gujarat, Rajasthan (west)
        "43N": "EPSG:32643",  # Maharashtra, Madhya Pradesh
        "44N": "EPSG:32644",  # Central India (default)
        "45N": "EPSG:32645",  # Bihar, Jharkhand, Odisha
        "46N": "EPSG:32646",  # Northeast India, Assam
    }

    @staticmethod
    def get_utm_zone(longitude: float) -> str:
        """Return appropriate UTM EPSG code for an Indian longitude."""
        zone_number = int((longitude + 180) / 6) + 1
        return f"EPSG:326{zone_number}"


# Global singleton
settings = Settings()
