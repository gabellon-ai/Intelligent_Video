"""
Analysis configuration and results endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import List, Optional
from pydantic import BaseModel

from ..config import settings

router = APIRouter()


class DetectionConfig(BaseModel):
    """Custom detection configuration"""
    queries: List[str] = settings.DEFAULT_QUERIES
    threshold: float = settings.CONFIDENCE_THRESHOLD
    sample_fps: float = settings.SAMPLE_FPS


class AnalysisRequest(BaseModel):
    """Request to start analysis with custom config"""
    job_id: str
    config: Optional[DetectionConfig] = None


@router.get("/default-config")
async def get_default_config():
    """Get default detection configuration"""
    return {
        "queries": settings.DEFAULT_QUERIES,
        "threshold": settings.CONFIDENCE_THRESHOLD,
        "sample_fps": settings.SAMPLE_FPS,
        "batch_size": settings.BATCH_SIZE
    }


@router.post("/configure/{job_id}")
async def configure_analysis(job_id: str, config: DetectionConfig):
    """
    Set custom detection parameters for a job
    Must be called before starting analysis
    """
    from .videos import jobs
    
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    
    job = jobs[job_id]
    if job.status != "pending":
        raise HTTPException(400, "Can only configure pending jobs")
    
    job.config = config.model_dump()
    
    return {"message": "Configuration updated", "config": job.config}


@router.get("/presets")
async def get_detection_presets():
    """Pre-configured detection presets for common scenarios"""
    return {
        "warehouse_general": {
            "name": "Warehouse - General",
            "queries": [
                "forklift",
                "pallet",
                "person",
                "cardboard box",
                "conveyor belt"
            ],
            "threshold": 0.15
        },
        "warehouse_safety": {
            "name": "Warehouse - Safety Focus",
            "queries": [
                "person",
                "person wearing safety vest",
                "person without hardhat",
                "forklift",
                "fallen person"
            ],
            "threshold": 0.12
        },
        "logistics_agv": {
            "name": "Logistics - AGV Tracking",
            "queries": [
                "AGV automated guided vehicle",
                "robot",
                "autonomous mobile robot",
                "pallet",
                "person"
            ],
            "threshold": 0.15
        },
        "loading_dock": {
            "name": "Loading Dock",
            "queries": [
                "truck",
                "trailer",
                "forklift",
                "pallet",
                "person",
                "open dock door"
            ],
            "threshold": 0.15
        }
    }
