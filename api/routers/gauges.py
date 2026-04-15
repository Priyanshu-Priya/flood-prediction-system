"""
FastAPI Gauge Data Router
============================
Endpoints for real-time and historical gauge data access.
Supports both India-WRIS and GloFAS (Copernicus EWDS) backends.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from loguru import logger

from api.dependencies import get_gauge_client
from api.schemas import GaugeReading

router = APIRouter(prefix="/gauges", tags=["Gauge Data"])


@router.get("/stations")
async def list_stations(
    state: str | None = Query(None, description="Filter by Indian state"),
    basin: str | None = Query(None, description="Filter by river basin"),
    date: str | None = Query(None, description="Optional target date for historical simulation (YYYY-MM-DD)"),
):
    """List available gauge stations (GloFAS or India-WRIS)."""
    client = get_gauge_client()
    stations_df = client.fetch_station_metadata(state=state, basin=basin)
    stations_list = stations_df.to_dict(orient="records")
    
    if date:
        from src.ingestion.offline_dataset import OfflineDataset
        import numpy as np
        
        # Batch extract discharge values for all stations on this date
        coords = [{"station_id": s["station_id"], "lat": s["lat"], "lon": s["lon"]} for s in stations_list]
        discharge_map = OfflineDataset.get_bulk_data_by_date(date, coords)
        
        for s in stations_list:
            sid = s["station_id"]
            q = discharge_map.get(sid, 0.0)
            s["discharge_cumecs"] = q
            
            # Simple simulation: map discharge to level relative to danger
            # Using a simplified 0.4 power law: level = c * Q^0.4
            # We assume q=2000 is approximately danger level for most stations in this dataset
            danger = s.get("danger_level_m", 15.0)
            warning = s.get("warning_level_m", 13.0)
            s["level"] = (q / 2000.0)**0.4 * danger if q > 0 else 0.0
            
            # Clamp to sensible min
            if s["level"] < 2.0 and q > 0:
                s["level"] = 2.0
                
            # Alert Logic
            if s["level"] >= danger:
                s["alert"] = "RED"
            elif s["level"] >= warning:
                s["alert"] = "ORANGE"
            elif s["level"] >= warning * 0.8:
                s["alert"] = "YELLOW"
            else:
                s["alert"] = "GREEN"
            
            s["timestamp"] = date

    return {"stations": stations_list, "count": len(stations_list)}


@router.get("/live/{station_id}", response_model=GaugeReading)
async def get_live_reading(station_id: str):
    """Get latest water level for a specific station."""
    client = get_gauge_client()

    try:
        data = client.fetch_realtime_levels([station_id])
    except Exception as e:
        raise HTTPException(500, f"Failed to fetch live data: {e}")

    if data.empty:
        raise HTTPException(404, f"No data for station {station_id}")

    row = data.iloc[0]
    return GaugeReading(
        station_id=station_id,
        station_name=row.get("name", ""),
        river=row.get("river", ""),
        lat=row.get("lat", 0),
        lon=row.get("lon", 0),
        timestamp=row.get("timestamp"),
        water_level_m=row.get("water_level_m", 0),
        discharge_cumecs=row.get("discharge_cumecs"),
        danger_level_m=row.get("danger_level_m"),
        warning_level_m=row.get("warning_level_m"),
        trend=row.get("trend"),
    )


@router.get("/historical/{station_id}")
async def get_historical_data(
    station_id: str,
    start_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)"),
    frequency: str = Query("daily", description="Data frequency"),
):
    """Get historical discharge / water level time series for a station."""
    client = get_gauge_client()

    try:
        data = client.fetch_water_levels(
            station_id, start_date, end_date, frequency
        )
        return {
            "station_id": station_id,
            "start_date": start_date,
            "end_date": end_date,
            "observations": data.reset_index().to_dict(orient="records"),
            "count": len(data),
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to fetch data: {e}")

