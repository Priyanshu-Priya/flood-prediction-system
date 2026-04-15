"""
FastAPI Application — Flood Risk Prediction Service
=====================================================
Main entry point for the prediction API.

Features:
- REST endpoints for LSTM and XGBoost predictions
- WebSocket endpoint for real-time flood alerts
- CORS middleware for Streamlit dashboard
- Rate limiting for public-facing endpoints
- Gzip compression for GeoTIFF responses
- Health check with model status

Run: uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from loguru import logger

from api.dependencies import get_lstm_model, get_xgboost_model, get_uptime, check_gpu
from api.routers import predictions, gauges, risk_maps
from api.schemas import HealthResponse
from config.logging_config import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — model preloading on startup."""
    setup_logging(level="INFO")
    logger.info("🌊 Flood Risk Prediction Service starting...")

    # Models are preloaded on demand or during first request to keep first startup fast
    # but here we trigger lazy loading to ensure they are ready in cache.
    lstm = get_lstm_model()
    xgb = get_xgboost_model()
    
    # Preload models into cache
    lstm = get_lstm_model()
    xgb = get_xgboost_model()

    logger.info(
        f"Models loaded | LSTM={'✓' if lstm else '✗'} | "
        f"XGBoost={'✓' if xgb else '✗'}"
    )

    yield

    logger.info("Service shutting down...")


app = FastAPI(
    title="🌊 Flood Risk Prediction API — India",
    description=(
        "Real-time flood forecasting and spatial susceptibility mapping "
        "for the Indian subcontinent. Powered by hybrid LSTM + XGBoost "
        "with Sentinel-1 SAR, ALOS PALSAR DEM, and India-WRIS data."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Middleware ──

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# ── Routers ──

app.include_router(predictions.router)
app.include_router(gauges.router)
app.include_router(risk_maps.router)


# ── Core Endpoints ──

@app.get("/", tags=["Root"])
async def root():
    """API landing page."""
    return {
        "service": "Flood Risk Prediction API — India",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "predict_water_level": "POST /predict/water-level",
            "predict_susceptibility": "POST /predict/susceptibility",
            "predict_combined": "POST /predict/combined",
            "list_stations": "GET /gauges/stations",
            "live_reading": "GET /gauges/live/{station_id}",
            "historical_data": "GET /gauges/historical/{station_id}",
            "risk_maps": "GET /risk-map/{region_id}",
            "health": "GET /health",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Service health check with model status."""
    gpu_available, gpu_name = check_gpu()

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        models_loaded={
            "lstm_forecaster": get_lstm_model() is not None,
            "xgboost_susceptibility": get_xgboost_model() is not None,
        },
        uptime_seconds=round(get_uptime(), 1),
        gpu_available=gpu_available,
        gpu_name=gpu_name,
    )


# ── WebSocket for Real-Time Alerts ──

class ConnectionManager:
    """Manages WebSocket connections for real-time flood alerts."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected | total={len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected | total={len(self.active_connections)}")

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass


alert_manager = ConnectionManager()


@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    """
    WebSocket endpoint for real-time flood alerts.

    Clients connect and receive push notifications when
    flood conditions change at any monitored station.
    """
    await alert_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive, listen for client messages
            data = await websocket.receive_text()
            logger.debug(f"WebSocket received: {data}")
    except WebSocketDisconnect:
        alert_manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        workers=4,
        reload=True,
        log_level="info",
    )
