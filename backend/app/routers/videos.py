"""
Video upload and management endpoints
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
import uuid
import os
import aiofiles
from pathlib import Path
from typing import Optional

from ..config import settings
from ..models.schemas import VideoUploadResponse, VideoStatus, AnalysisJob
from ..services.analysis_runner import AnalysisRunner

router = APIRouter()

# In-memory job tracking (use Redis in production)
jobs: dict[str, AnalysisJob] = {}


@router.post("/upload", response_model=VideoUploadResponse)
async def upload_video(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Upload a video file for analysis
    Returns job_id to track progress via WebSocket
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(400, "File must be a video")
    
    # Generate unique ID
    job_id = str(uuid.uuid4())
    filename = f"{job_id}_{file.filename}"
    filepath = Path(settings.UPLOAD_DIR) / filename
    
    # Stream file to disk
    async with aiofiles.open(filepath, 'wb') as out_file:
        while chunk := await file.read(1024 * 1024):  # 1MB chunks
            await out_file.write(chunk)
    
    # Create job record
    job = AnalysisJob(
        job_id=job_id,
        filename=file.filename,
        filepath=str(filepath),
        status=VideoStatus.PENDING
    )
    jobs[job_id] = job
    
    return VideoUploadResponse(
        job_id=job_id,
        filename=file.filename,
        status=VideoStatus.PENDING,
        message="Video uploaded. Connect to WebSocket for progress."
    )


@router.get("/{job_id}/status")
async def get_job_status(job_id: str):
    """Get current status of analysis job"""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    
    job = jobs[job_id]
    return {
        "job_id": job.job_id,
        "status": job.status,
        "progress": job.progress,
        "frames_processed": job.frames_processed,
        "total_frames": job.total_frames,
        "summary": job.summary
    }


@router.get("/{job_id}/results")
async def get_job_results(job_id: str):
    """Get full analysis results"""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    
    job = jobs[job_id]
    if job.status != VideoStatus.COMPLETED:
        raise HTTPException(400, f"Job not complete. Status: {job.status}")
    
    return {
        "job_id": job.job_id,
        "summary": job.summary,
        "timeline": job.timeline,
        "output_video": job.output_path
    }


@router.delete("/{job_id}")
async def delete_job(job_id: str):
    """Delete job and associated files"""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    
    job = jobs[job_id]
    
    # Clean up files
    if job.filepath and os.path.exists(job.filepath):
        os.remove(job.filepath)
    if job.output_path and os.path.exists(job.output_path):
        os.remove(job.output_path)
    
    del jobs[job_id]
    return {"message": "Job deleted"}


def get_jobs_store():
    """Dependency to access jobs store"""
    return jobs
