"""Tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


class TestHealthEndpoint:
    def test_health_check(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "models_loaded" in data
        assert "uptime_seconds" in data

    def test_root_endpoint(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "endpoints" in data


class TestPredictionEndpoints:
    def test_water_level_prediction_no_model(self):
        """When model isn't loaded, should return 503."""
        response = client.post(
            "/predict/water-level",
            json={"station_id": "CWC003", "forecast_hours": 48},
        )
        # 503 if model not trained, 200 if model exists
        assert response.status_code in [200, 503]

    def test_susceptibility_prediction_no_model(self):
        response = client.post(
            "/predict/susceptibility",
            json={
                "min_lon": 80.0, "min_lat": 12.8,
                "max_lon": 80.4, "max_lat": 13.2,
            },
        )
        assert response.status_code in [200, 503]


class TestGaugeEndpoints:
    def test_list_stations(self):
        response = client.get("/gauges/stations")
        assert response.status_code == 200
        data = response.json()
        assert "stations" in data
        assert "count" in data

    def test_list_stations_filtered(self):
        response = client.get("/gauges/stations?basin=Ganga")
        assert response.status_code == 200


class TestRiskMapEndpoints:
    def test_list_maps(self):
        response = client.get("/risk-map/")
        assert response.status_code == 200
        assert "maps" in response.json()

    def test_missing_map(self):
        response = client.get("/risk-map/nonexistent_region")
        assert response.status_code == 404
