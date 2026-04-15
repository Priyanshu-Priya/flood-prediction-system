"""
Engine — Core data models for the Flood Risk Prediction System.
"""

from pydantic import BaseModel


class AreaOfInterest(BaseModel):
    """Defines the geographic area of interest for predictions."""
    name: str = "India"
    geojson_data: dict | None = None
    bbox: tuple[float, float, float, float] = (68.0, 6.0, 98.0, 36.0)
