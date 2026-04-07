"""
Pydantic Data Validation Schemas
==================================
Strict data contracts for all pipeline interfaces.
Ensures data integrity at module boundaries.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class BoundingBox(BaseModel):
    """WGS84 bounding box for AOI definition."""
    min_lon: float = Field(..., ge=-180, le=180, description="Western boundary")
    min_lat: float = Field(..., ge=-90, le=90, description="Southern boundary")
    max_lon: float = Field(..., ge=-180, le=180, description="Eastern boundary")
    max_lat: float = Field(..., ge=-90, le=90, description="Northern boundary")

    @field_validator("max_lon")
    @classmethod
    def validate_lon_range(cls, v, info):
        if "min_lon" in info.data and v <= info.data["min_lon"]:
            raise ValueError("max_lon must be greater than min_lon")
        return v

    @field_validator("max_lat")
    @classmethod
    def validate_lat_range(cls, v, info):
        if "min_lat" in info.data and v <= info.data["min_lat"]:
            raise ValueError("max_lat must be greater than min_lat")
        return v

    def to_tuple(self) -> tuple[float, float, float, float]:
        return (self.min_lon, self.min_lat, self.max_lon, self.max_lat)


class GaugeObservation(BaseModel):
    """Single gauge observation record."""
    station_id: str
    timestamp: datetime
    water_level_m: Optional[float] = None
    discharge_cumecs: Optional[float] = None
    quality_flag: str = "good"

    @field_validator("water_level_m")
    @classmethod
    def validate_water_level(cls, v):
        if v is not None and (v < -50 or v > 500):
            raise ValueError(f"Water level {v}m is physically unreasonable")
        return v


class FloodPrediction(BaseModel):
    """LSTM flood forecast output."""
    station_id: str
    forecast_timestamp: datetime
    lead_time_hours: int
    water_level_mean_m: float
    water_level_std_m: float
    lower_ci_90_m: float
    upper_ci_90_m: float
    alert_level: str = Field(..., pattern="^(GREEN|YELLOW|ORANGE|RED)$")


class SusceptibilityCell(BaseModel):
    """Single grid cell susceptibility result."""
    lat: float
    lon: float
    flood_probability: float = Field(..., ge=0, le=1)
    risk_class: str = Field(..., pattern="^(GREEN|YELLOW|ORANGE|RED)$")
    twi: Optional[float] = None
    slope_degrees: Optional[float] = None


class PipelineConfig(BaseModel):
    """Configuration for a data processing pipeline run."""
    aoi_name: str
    bbox: BoundingBox
    start_date: str
    end_date: str
    dem_source: str = "alos_palsar"
    target_crs: Optional[str] = None
    compute_features: list[str] = Field(
        default=["twi", "slope", "flow_accumulation", "distance_to_channel"]
    )
