"""
Video upload and management endpoints
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
import asyncio
import logging
import uuid
import os
import aiofiles
from pathlib import Path
from typing import Optional

from ..config import settings
from ..models.schemas import VideoUploadResponse, VideoStatus, AnalysisJob

logger = logging.getLogger(__name__)
router = APIRouter()


async def _probe_video_codec(filepath: Path) -> str:
    """Return the video codec name (e.g. 'h264', 'hevc') or '' if unknown."""
    proc = await asyncio.create_subprocess_exec(
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name",
        "-of", "default=nw=1:nk=1",
        str(filepath),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    out, _ = await proc.communicate()
    return (out or b"").decode().strip().lower()


async def _prepare_for_browser(filepath: Path) -> None:
    """Make the uploaded mp4 browser-playable.

    1. If the video codec is HEVC (H.265), Chromium on Linux refuses to
       decode it in <video>. Transcode to H.264 via GPU NVENC.
    2. Otherwise, ensure the moov atom is at the front of the file so the
       <video> element can start progressive playback without downloading
       the whole file.
    """
    codec = await _probe_video_codec(filepath)
    logger.info(f"uploaded codec={codec} for {filepath}")

    if codec in {"hevc", "h265"}:
        tmp_path = filepath.with_suffix(filepath.suffix + ".transcoding")
        logger.info(f"transcoding HEVC -> H.264 via NVENC: {filepath}")
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y",
            "-hwaccel", "cuda",
            "-i", str(filepath),
            "-c:v", "h264_nvenc", "-preset", "p5", "-cq", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            "-f", "mp4",
            str(tmp_path),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, err = await proc.communicate()
        rc = proc.returncode
        if rc == 0:
            tmp_path.replace(filepath)
            logger.info(f"transcode ok: {filepath}")
        else:
            err_tail = (err or b"").decode(errors="replace").splitlines()[-5:]
            logger.warning(f"NVENC transcode rc={rc} for {filepath}: {err_tail}")
            try: tmp_path.unlink(missing_ok=True)
            except Exception: pass
        return

    # Already H.264 (or other browser-friendly codec). Only need faststart.
    try:
        with open(filepath, "rb") as f:
            head = f.read(4096)
        if b"moov" in head:
            return
    except Exception as e:
        logger.warning(f"faststart probe failed for {filepath}: {e}")
        return

    tmp_path = filepath.with_suffix(filepath.suffix + ".faststart")
    logger.info(f"faststart rewrite: {filepath}")
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-y", "-i", str(filepath),
        "-c", "copy", "-movflags", "+faststart",
        str(tmp_path),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    rc = await proc.wait()
    if rc == 0:
        tmp_path.replace(filepath)
        logger.info(f"faststart ok: {filepath}")
    else:
        logger.warning(f"faststart ffmpeg rc={rc} for {filepath}")
        try: tmp_path.unlink(missing_ok=True)
        except Exception: pass

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

    # Make the mp4 browser-playable (H.264 + moov-at-front).
    await _prepare_for_browser(filepath)

    # Canonical <uuid>.mp4 hardlink so nginx can serve the file directly
    # without needing to glob on the original filename. nginx bypasses
    # FastAPI for video bytes so scrubbing doesn't starve the inference loop.
    canonical = Path(settings.UPLOAD_DIR) / f"{job_id}.mp4"
    try:
        if canonical.exists():
            canonical.unlink()
        os.link(filepath, canonical)
    except OSError as e:
        logger.warning(f"Could not hardlink canonical mp4 for {job_id}: {e}")

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


@router.get("/{job_id}/video")
async def get_video_file(job_id: str):
    """Get the uploaded video file path for playback"""
    from fastapi.responses import FileResponse

    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    job = jobs[job_id]
    if not job.filepath or not os.path.exists(job.filepath):
        raise HTTPException(404, "Video file not found")

    return FileResponse(
        job.filepath,
        media_type="video/mp4",
        filename=job.filename
    )


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
