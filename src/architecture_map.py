import ee
from enum import Enum
from pathlib import Path
from pydantic import BaseModel, Field

class DataLayer(str, Enum):
    IMD = "imd"
    GRDC = "grdc"
    GLOFAS = "glofas"
    GEE_S1 = "gee_sentinel_1"

class AOI(BaseModel):
    name: str = Field(default="India")
    bbox: tuple[float, float, float, float] = Field(default=(68.0, 6.0, 98.0, 36.0))
    geojson: dict | None = None

class DataIngestionLayer:
    def __init__(self, use_global_fallback: bool = True):
        self.use_global_fallback = use_global_fallback
        self.active_sources = [DataLayer.IMD, DataLayer.GEE_S1]
        if self.use_global_fallback:
            self.active_sources.extend([DataLayer.GRDC, DataLayer.GLOFAS])

class FeatureEngineering:
    pass

class MachineLearningModels:
    pass

class HybridSystemArchitecture:
    def __init__(self):
        self.ingestion = DataIngestionLayer()
        self.features = FeatureEngineering()
        self.models = MachineLearningModels()
        
    def authenticate_gee(self, project_id: str, sa_path: Path):
        ee.Initialize(project=project_id)
        
    def dispatch_source(self, region_is_india: bool) -> list[DataLayer]:
        if region_is_india:
            return [DataLayer.GLOFAS, DataLayer.IMD]
        return [DataLayer.GRDC, DataLayer.GLOFAS]
