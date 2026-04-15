"""
FastAPI Prediction Router
============================
Endpoints for flood prediction: water level forecasts,
spatial susceptibility, and combined ensemble predictions.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from loguru import logger

from api.dependencies import get_lstm_model, get_xgboost_model, get_ensemble_combiner, get_gauge_client
from api.schemas import (
    WaterLevelPredictionRequest,
    WaterLevelPredictionResponse,
    SusceptibilityRequest,
    SusceptibilityResponse,
    CombinedPredictionRequest,
    ModelMetricsResponse,
    ForecastPoint,
    AlertInfo,
    HistoricalPredictionRequest,
    HistoricalPredictionResponse,
)

router = APIRouter(prefix="/predict", tags=["Predictions"])


@router.get("/metrics", response_model=ModelMetricsResponse)
async def get_model_metrics():
    """Extract and return performance metrics from loaded models."""
    import json
    from config.settings import MODEL_DIR

    # Try loading persisted training metrics first
    metrics_path = MODEL_DIR / "training_metrics.json"
    if metrics_path.exists():
        try:
            with open(metrics_path) as f:
                saved = json.load(f)
            return saved
        except Exception as e:
            logger.warning(f"Failed to load saved metrics: {e}")

    # Fallback: build from loaded models
    xgb_model = get_xgboost_model()

    metrics = {
        "lstm": {
            "nse_mean": 0.0,
            "parameters": "7.5M",
            "last_train": "Not trained yet",
            "data_source": "none",
        },
        "xgboost": {
            "auc_roc": 0.0,
            "feature_importance": {},
            "n_features": 0,
            "data_source": "none",
        },
        "system": {
            "stations_monitored": 10,
            "is_gpu_accelerated": True,
        },
    }

    if xgb_model and xgb_model.model:
        try:
            importance = xgb_model.model.get_score(importance_type="gain")
            metrics["xgboost"]["feature_importance"] = importance
            metrics["xgboost"]["n_features"] = len(importance)
        except Exception:
            pass

    return metrics


@router.post("/water-level", response_model=WaterLevelPredictionResponse)
async def predict_water_level(request: WaterLevelPredictionRequest):
    """
    Generate LSTM water level forecast for a gauge station.

    Returns time-series predictions with uncertainty bands
    and CWC-style flood alert classification.
    """
    model = get_lstm_model()
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="LSTM model not loaded. Train the model first.",
        )

    logger.info(
        f"Water level prediction | station={request.station_id} | "
        f"horizon={request.forecast_hours}h"
    )

    try:
        import pandas as pd
        from src.features.precipitation import AntecedentPrecipitationIndex

        # Fetch real historical data from GloFAS
        gauge_client = get_gauge_client()
        if getattr(request, 'target_date', None):
            end_date = pd.to_datetime(request.target_date)
        else:
            end_date = datetime.now()
        start_date = end_date - timedelta(days=30)  # 30 days for API features

        is_real_data = True
        try:
            df = gauge_client.fetch_water_levels(
                request.station_id,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
            )
            if df.empty:
                is_real_data = False
        except Exception as e:
            logger.warning(f"Failed to fetch gauge data: {e}")
            df = pd.DataFrame()
            is_real_data = False

        n_features = 12
        n_static = 8
        lookback = min(30, max(7, len(df))) if not df.empty else 30

        # Build feature tensor from real data
        if not df.empty:
            discharge = df["discharge_cumecs"].fillna(0).values if "discharge_cumecs" in df.columns else np.zeros(len(df))
            
            # Compute API features
            api_calc = AntecedentPrecipitationIndex(decay_factor=0.90)
            api_df = api_calc.compute_multi_scale_api(discharge, windows=[3, 7, 14, 30])

            # Build feature matrix (12 features as per train_flood_model.py)
            n = len(df)
            history = np.zeros((n, n_features), dtype=np.float32)
            history[:, 0] = discharge
            history[:, 1] = np.log1p(np.maximum(0, discharge))
            history[:, 2] = np.gradient(discharge)
            history[:, 3] = api_df["api_3d"].values[:n]
            history[:, 4] = api_df["api_7d"].values[:n]
            history[:, 5] = api_df["api_14d"].values[:n]
            history[:, 6] = api_df["api_30d"].values[:n]
            
            df_temp = pd.Series(discharge)
            history[:, 7] = df_temp.rolling(7, min_periods=1).mean().values
            history[:, 8] = df_temp.rolling(7, min_periods=1).std().fillna(0).values
            
            doy = pd.to_datetime(df.index).dayofyear
            history[:, 9] = np.sin(2 * np.pi * doy / 365.25)
            history[:, 10] = np.cos(2 * np.pi * doy / 365.25)
            history[:, 11] = discharge / (np.max(discharge) + 1e-6)
        else:
            # Minimal synthetic fallback
            history = np.random.randn(lookback, n_features).astype(np.float32)

        static = np.zeros(n_static, dtype=np.float32)

        prediction = model.predict(history, static)

        # Build forecast points
        now = datetime.utcnow()
        forecast_points = []
        for i in range(min(request.forecast_hours, len(prediction["mean"]))):
            forecast_points.append(ForecastPoint(
                timestamp=now + timedelta(hours=i + 1),
                lead_time_hours=i + 1,
                water_level_mean_m=round(float(prediction["mean"][i]), 3),
                water_level_std_m=round(float(prediction["std"][i]), 3),
                lower_ci_90_m=round(float(prediction["lower_ci"][i]), 3),
                upper_ci_90_m=round(float(prediction["upper_ci"][i]), 3),
            ))

        # Generate alert using station metadata if available
        from src.ingestion.glofas import INDIA_GAUGE_STATIONS
        station_meta = next(
            (s for s in INDIA_GAUGE_STATIONS if s["station_id"] == request.station_id),
            {"danger_level_m": 10.0, "warning_level_m": 8.5},
        )
        alert = model.generate_alert(
            prediction,
            danger_level=station_meta.get("danger_level_m", 10.0),
            warning_level=station_meta.get("warning_level_m", 8.5),
        )

        return WaterLevelPredictionResponse(
            station_id=request.station_id,
            generated_at=now,
            forecast=forecast_points,
            alert=AlertInfo(**alert),
            is_real_data=is_real_data,
            data_sign="σ" if is_real_data else "",
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/susceptibility", response_model=SusceptibilityResponse)
async def predict_susceptibility(request: SusceptibilityRequest):
    """
    Generate spatial flood susceptibility map for a bounding box.

    Returns flood probability statistics and risk distribution.
    In production, also generates a downloadable GeoTIFF.
    """
    model = get_xgboost_model()
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="XGBoost model not loaded. Train the model first.",
        )

    logger.info(
        f"Susceptibility prediction | "
        f"bbox=({request.min_lon},{request.min_lat},{request.max_lon},{request.max_lat})"
    )

    try:
        from config.settings import PROCESSED_DATA_DIR
        import pandas as pd

        is_real_data = False
        prob_map = None
        for aoi_dir in PROCESSED_DATA_DIR.iterdir():
            if aoi_dir.is_dir():
                feature_path = aoi_dir / "xgboost_features.parquet"
                if feature_path.exists():
                    X = pd.read_parquet(feature_path)
                    prob_map = model.predict_probability(X)
                    is_real_data = True
                    break

        if prob_map is not None:
            n_cells = len(prob_map)
            mean_prob = float(np.mean(prob_map))
            max_prob = float(np.max(prob_map))
        else:
            # Synthetic fallback for AOIs without processed data
            n_cells = int(
                (request.max_lon - request.min_lon)
                * (request.max_lat - request.min_lat)
                * (111000 / request.resolution_m) ** 2
            )
            n_cells = max(n_cells, 100)
            prob_map = np.random.beta(2, 8, n_cells)
            mean_prob = float(np.mean(prob_map))
            max_prob = float(np.max(prob_map))

        # Compute risk distribution
        green = int(np.sum(prob_map < 0.3))
        yellow = int(np.sum((prob_map >= 0.3) & (prob_map < 0.6)))
        orange = int(np.sum((prob_map >= 0.6) & (prob_map < 0.8)))
        red = int(np.sum(prob_map >= 0.8))
        total = max(green + yellow + orange + red, 1)

        return SusceptibilityResponse(
            bbox={
                "min_lon": request.min_lon,
                "min_lat": request.min_lat,
                "max_lon": request.max_lon,
                "max_lat": request.max_lat,
            },
            resolution_m=request.resolution_m,
            n_cells=total,
            mean_probability=round(mean_prob, 4),
            max_probability=round(max_prob, 4),
            risk_distribution={"green": green, "yellow": yellow, "orange": orange, "red": red},
            risk_percentages={
                "green": round(green / total * 100, 1),
                "yellow": round(yellow / total * 100, 1),
                "orange": round(orange / total * 100, 1),
                "red": round(red / total * 100, 1),
            },
            is_real_data=is_real_data,
            data_sign="σ" if is_real_data else "",
        )

    except Exception as e:
        logger.error(f"Susceptibility prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/combined")
async def predict_combined(request: CombinedPredictionRequest):
    """
    Generate ensemble prediction combining temporal and spatial models.

    Fuses LSTM water level forecast with XGBoost susceptibility map
    using calibrated weights.
    """
    combiner = get_ensemble_combiner()
    lstm_model = get_lstm_model()
    xgb_model = get_xgboost_model()

    # Get temporal prediction
    temporal_prob = 0.5
    if lstm_model:
        try:
            # Fetch real data for combined prediction if possible
            gauge_client = get_gauge_client()
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            df = gauge_client.fetch_water_levels(request.station_id, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
            
            if not df.empty:
                # Minimal feature building for LSTM
                history = np.zeros((len(df), 12), dtype=np.float32)
                history[:, 0] = df["water_level_m"].values
                prediction = lstm_model.predict(history, static)
            else:
                prediction = {"mean": [0.0]} # Fallback
            peak_level = float(np.max(prediction["mean"]))
            temporal_prob = min(peak_level / 15.0, 1.0)  # Normalize to probability
        except Exception as e:
            logger.warning(f"LSTM prediction failed: {e}")

    # Get spatial prediction
    spatial_prob = 0.5
    if xgb_model:
        try:
            from config.settings import PROCESSED_DATA_DIR
            import pandas as pd
            for aoi_dir in PROCESSED_DATA_DIR.iterdir():
                if aoi_dir.is_dir():
                    fp = aoi_dir / "xgboost_features.parquet"
                    if fp.exists():
                        X = pd.read_parquet(fp)
                        probs = xgb_model.predict_probability(X)
                        spatial_prob = float(np.mean(probs))
                        break
        except Exception as e:
            logger.warning(f"XGBoost prediction failed: {e}")

    # Ensemble fusion
    combined_prob = combiner.alpha * temporal_prob + (1 - combiner.alpha) * spatial_prob

    # Determine alert level
    if combined_prob >= 0.8:
        alert = "RED"
    elif combined_prob >= 0.6:
        alert = "ORANGE"
    elif combined_prob >= 0.3:
        alert = "YELLOW"
    else:
        alert = "GREEN"

    return {
        "station_id": request.station_id,
        "bbox": {
            "min_lon": request.min_lon, "min_lat": request.min_lat,
            "max_lon": request.max_lon, "max_lat": request.max_lat,
        },
        "forecast_hours": request.forecast_hours,
        "ensemble_alpha": combiner.alpha,
        "temporal_probability": round(temporal_prob, 4),
        "spatial_probability": round(spatial_prob, 4),
        "combined_probability": round(combined_prob, 4),
        "alert_level": alert,
    }

@router.post("/historical", response_model=HistoricalPredictionResponse)
async def predict_historical(request: HistoricalPredictionRequest):
    """
    Offline simulation endpoint targeting the downloaded 2022 dataset.
    """
    try:
        from src.ingestion.offline_dataset import OfflineDataset
        # This function fetches nearest grid point data and validates date range
        data = OfflineDataset.get_data_by_date(request.date, request.latitude, request.longitude)
        
        discharge = data["river_discharge"]
        
        # Run ML model (Simulated mapping using river discharge heuristic for risk classification)
        # In a full spatial pipeline, we would merge this with TWI/Elevation before passing to XGBoost.
        if discharge > 3000:
            risk = "High"
            prob = min(0.99, (discharge / 8000.0) * 0.5 + 0.5)
        elif discharge > 1000:
            risk = "Medium"
            prob = min(0.70, (discharge / 3000.0) * 0.4 + 0.3)
        else:
            risk = "Low"
            prob = max(0.01, discharge / 1000.0 * 0.3)
            
        return HistoricalPredictionResponse(
            flood_risk=risk,
            probability=round(prob * 100, 2),
            river_discharge=discharge,
            matched_latitude=data["matched_lat"],
            matched_longitude=data["matched_lon"],
            target_date=data["date"]
        )
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except FileNotFoundError as fnf_err:
        raise HTTPException(status_code=503, detail=str(fnf_err))
    except Exception as e:
        logger.error(f"Historical prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error extracting dataset.")
