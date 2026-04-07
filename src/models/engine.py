import torch
import torch.nn as nn
from pydantic import BaseModel
import xgboost as xgb
import numpy as np

class AreaOfInterest(BaseModel):
    name: str = "India"
    geojson_data: dict | None = None
    bbox: tuple[float, float, float, float] = (68.0, 6.0, 98.0, 36.0)

class TimeSeriesLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.cuda.amp.autocast():
            lstm_out, _ = self.lstm(x)
            out = self.linear(lstm_out[:, -1, :])
            return out

class SpatialRiskClassifier:
    def __init__(self):
        self.model = None

    def train(self, dtrain: xgb.DMatrix, parameters: dict, num_rounds: int = 100):
        params = parameters.copy()
        params.update({"tree_method": "gpu_hist", "device": "cuda"})
        self.model = xgb.train(params, dtrain, num_rounds)

    def predict(self, dmatrix: xgb.DMatrix) -> np.ndarray:
        return self.model.predict(dmatrix)
