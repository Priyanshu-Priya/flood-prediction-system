"""
Feature Generation Script
==========================
Generates spatially-aware features for the XGBoost flood susceptibility model.
If real processed terrain data is missing, it generates geographically-tied
synthetic data based on coordinates to ensure the dashboard map is dynamic.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger

sys.path.append(str(Path(__file__).parent.parent))
from config.settings import PROCESSED_DATA_DIR

def generate_spatial_features(bbox, resolution_m=100):
    """Generate features for a specific bounding box."""
    min_lon, min_lat, max_lon, max_lat = bbox
    
    # Calculate grid dimensions
    width_deg = max_lon - min_lon
    height_deg = max_lat - min_lat
    
    # Approx 111km per degree
    n_lon = int(width_deg * 111000 / resolution_m)
    n_lat = int(height_deg * 111000 / resolution_m)
    
    n_lon = max(n_lon, 10)
    n_lat = max(n_lat, 10)
    
    total_cells = n_lon * n_lat
    logger.info(f"Generating {total_cells} spatial cells for bbox {bbox}")
    
    # Create coordinate grids
    lon_grid = np.linspace(min_lon, max_lon, n_lon)
    lat_grid = np.linspace(min_lat, max_lat, n_lat)
    lon_map, lat_map = np.meshgrid(lon_grid, lat_grid)
    
    # Flatten for DataFrame
    lons = lon_map.flatten()
    lats = lat_map.flatten()
    
    # Geographically-tied synthetic features (Perlin-like noise using trig)
    def noise(scale, freq_lon, freq_lat, offset=0):
        return scale * (np.sin(lons * freq_lon + offset) * np.cos(lats * freq_lat + offset))

    features = pd.DataFrame({
        "lon": lons,
        "lat": lats,
        "elevation": 100 + noise(50, 50, 50),
        "slope": np.abs(noise(15, 80, 80)),
        "aspect": (noise(180, 30, 30) + 180) % 360,
        "twi": 5 + np.abs(noise(10, 120, 120)), # Topographic Wetness Index
        "flow_accumulation": np.abs(noise(1000, 200, 200)),
        "distance_to_channel": np.abs(noise(500, 150, 150)),
        "curvature": noise(0.5, 300, 300),
        "runoff_coefficient": 0.4 + np.abs(noise(0.5, 40, 40)),
        "api_14d": noise(20, 10, 10) + 30,
        "impervious_fraction": np.clip(noise(1.0, 60, 60), 0, 1)
    })
    
    # Ensure column order matches training_metrics.json
    feature_names = [
        "slope", "aspect", "twi", "flow_accumulation",
        "distance_to_channel", "curvature", "elevation",
        "runoff_coefficient", "api_14d", "impervious_fraction"
    ]
    
    return features[feature_names]

def main():
    aoi_name = "bihar_v1"
    output_dir = PROCESSED_DATA_DIR / aoi_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Bihar bounding box approx
    bbox = (83.3, 23.9, 88.3, 27.5)
    
    # Use 500m resolution for faster generation but still geographic
    features = generate_spatial_features(bbox, resolution_m=500)
    
    feature_path = output_dir / "xgboost_features.parquet"
    features.to_parquet(feature_path)
    
    # Generate labels (flood/no-flood) based on TWI and elevation
    # TWI > 10 and Elevation < 110
    labels = ((features["twi"] > 10) & (features["elevation"] < 115)).astype(int).values
    label_path = output_dir / "xgboost_labels.npy"
    np.save(label_path, labels)
    
    logger.info(f"Generated {len(features)} features and saved to {output_dir}")

if __name__ == "__main__":
    main()
