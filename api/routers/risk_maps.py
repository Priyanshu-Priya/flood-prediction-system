"""
FastAPI Risk Maps Router
============================
Endpoints for serving pre-computed flood risk maps.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from loguru import logger

from config.settings import PROCESSED_DATA_DIR

router = APIRouter(prefix="/risk-map", tags=["Risk Maps"])


@router.get("/{region_id}")
async def get_risk_map(region_id: str):
    """
    Retrieve pre-computed flood risk map for a region.

    Returns the GeoTIFF file for client-side rendering.
    """
    map_path = PROCESSED_DATA_DIR / f"risk_map_{region_id}.tif"

    if not map_path.exists():
        raise HTTPException(404, f"Risk map not found for region: {region_id}")

    return FileResponse(
        str(map_path),
        media_type="image/tiff",
        filename=f"flood_risk_{region_id}.tif",
    )


@router.get("/")
async def list_available_maps():
    """List all available pre-computed risk maps."""
    maps_dir = PROCESSED_DATA_DIR
    if not maps_dir.exists():
        return {"maps": [], "count": 0}

    tif_files = list(maps_dir.glob("risk_map_*.tif"))
    maps = [
        {
            "region_id": f.stem.replace("risk_map_", ""),
            "filename": f.name,
            "size_mb": round(f.stat().st_size / 1e6, 2),
        }
        for f in tif_files
    ]

    return {"maps": maps, "count": len(maps)}
