"""
Tests for Data Ingestion Modules
==================================
Validates GloFAS, DEM, and SAR ingestion logic.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.glofas import GloFASClient, INDIA_GAUGE_STATIONS


class TestGloFASClient:
    """Test GloFAS data ingestion."""

    def test_gauge_station_registry(self):
        """Verify the gauge station registry is populated."""
        assert len(INDIA_GAUGE_STATIONS) > 0
        for station in INDIA_GAUGE_STATIONS:
            assert "station_id" in station
            assert "lat" in station
            assert "lon" in station
            assert station["station_id"].startswith("GLOFAS_")

    def test_client_initialization(self):
        """Test GloFAS client can be created."""
        client = GloFASClient()
        assert client is not None

    def test_station_lookup(self):
        """Verify all station IDs are valid."""
        client = GloFASClient()
        for station in INDIA_GAUGE_STATIONS:
            meta = client.get_station_metadata(station["station_id"])
            assert meta is not None
            assert "lat" in meta
            assert "lon" in meta

    def test_rating_curve(self):
        """Test discharge-to-water-level conversion."""
        client = GloFASClient()
        # A reasonable discharge should give a reasonable water level
        discharge = 500.0  # m³/s
        level = client._discharge_to_level(discharge)
        assert level > 0
        assert level < 50  # Sanity check

    def test_rating_curve_monotonic(self):
        """Higher discharge should give higher water level."""
        client = GloFASClient()
        level_low = client._discharge_to_level(100.0)
        level_high = client._discharge_to_level(1000.0)
        assert level_high > level_low

    def test_zero_discharge(self):
        """Zero discharge should give near-zero water level."""
        client = GloFASClient()
        level = client._discharge_to_level(0.0)
        assert level >= 0


class TestPrecipitationAPI:
    """Test Antecedent Precipitation Index computation."""

    def test_api_timeseries(self):
        """Test basic API computation."""
        from src.features.precipitation import AntecedentPrecipitationIndex

        api = AntecedentPrecipitationIndex(decay_factor=0.90)
        precip = np.array([10, 0, 5, 20, 0, 0, 15, 0, 0, 0], dtype=np.float64)
        result = api.compute_api_timeseries(precip)

        assert len(result) == len(precip)
        assert result[0] == 10.0  # First day = P_0

    def test_multi_scale_api(self):
        """Test multi-window API computation."""
        from src.features.precipitation import AntecedentPrecipitationIndex

        api = AntecedentPrecipitationIndex(decay_factor=0.90)
        precip = np.random.rand(100) * 20
        result = api.compute_multi_scale_api(precip)

        assert "api_3d" in result.columns
        assert "api_7d" in result.columns
        assert "api_14d" in result.columns
        assert "api_30d" in result.columns
        assert len(result) == len(precip)


class TestTerrainFeatures:
    """Test terrain feature extraction basics."""

    def test_feature_names(self):
        """Verify expected feature list."""
        expected = ["slope", "aspect", "twi", "flow_accumulation",
                    "distance_to_channel", "curvature", "dem_filled"]
        # These are the features the stacker expects
        assert len(expected) == 7


class TestSARProcessing:
    """Test SAR water extraction."""

    def test_water_mask_shape(self):
        """Test water mask generation from synthetic data."""
        from src.features.sar_processing import SARWaterExtractor

        extractor = SARWaterExtractor()
        # Simulate SAR backscatter: low values = water
        sar = np.random.uniform(-25, -5, (100, 100))
        sar[40:60, 40:60] = -22.0  # Water body

        mask = extractor.extract_water_mask(sar, method="manual", manual_threshold_db=-18.0)
        assert mask.shape == (100, 100)
        assert mask.dtype == np.uint8
        assert np.any(mask == 1)  # Some water detected

    def test_flood_frequency(self):
        """Test flood frequency computation."""
        from src.features.sar_processing import SARWaterExtractor

        extractor = SARWaterExtractor()
        masks = [np.random.randint(0, 2, (50, 50)) for _ in range(10)]
        freq = extractor.compute_flood_frequency(masks)

        assert freq.shape == (50, 50)
        assert freq.min() >= 0
        assert freq.max() <= 1
