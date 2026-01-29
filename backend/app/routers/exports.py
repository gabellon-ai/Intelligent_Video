"""
Export endpoints for downloading analysis results
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import io
from typing import Optional

from ..services.export_service import export_service
from .videos import jobs  # Import the jobs store


router = APIRouter()


def get_job_data(job_id: str):
    """Helper to get and validate job data"""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    
    job = jobs[job_id]
    
    if job.status != "completed":
        raise HTTPException(400, f"Job not completed. Current status: {job.status}")
    
    if not job.timeline:
        raise HTTPException(400, "No analysis data available for export")
    
    return job


@router.get("/{job_id}/export/csv")
async def export_csv(job_id: str):
    """
    Export frame-by-frame detection data as CSV
    
    Returns a downloadable CSV file with columns:
    - frame_number, timestamp_seconds, class_name, confidence, bbox coordinates
    """
    job = get_job_data(job_id)
    
    csv_content = export_service.generate_csv(
        job_id=job_id,
        timeline=job.timeline,
        summary=job.summary or {},
        video_info=None  # TODO: Store video_info in job
    )
    
    # Create streaming response
    return StreamingResponse(
        io.StringIO(csv_content),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="analysis_{job_id[:8]}.csv"'
        }
    )


@router.get("/{job_id}/export/json")
async def export_json(job_id: str):
    """
    Export full analysis data as JSON
    
    Returns comprehensive analysis data including:
    - Video info, summary statistics, timeline, and calculated statistics
    """
    job = get_job_data(job_id)
    
    json_data = export_service.generate_json(
        job_id=job_id,
        timeline=job.timeline,
        summary=job.summary or {},
        video_info=None,  # TODO: Store video_info in job
        filename=job.filename
    )
    
    # Return as downloadable JSON file
    return JSONResponse(
        content=json_data,
        headers={
            "Content-Disposition": f'attachment; filename="analysis_{job_id[:8]}.json"'
        }
    )


@router.get("/{job_id}/export/pdf")
async def export_pdf(job_id: str):
    """
    Export comprehensive PDF report
    
    Report includes:
    - Header with job info, date, duration
    - Summary statistics (total detections, by class)
    - Bar chart of object counts
    - Timeline visualization
    - Key frames with highest detection counts (annotated)
    """
    job = get_job_data(job_id)
    
    # Get video path for extracting key frames
    video_path = job.filepath if hasattr(job, 'filepath') else None
    
    try:
        pdf_bytes = export_service.generate_pdf(
            job_id=job_id,
            timeline=job.timeline,
            summary=job.summary or {},
            video_info=None,  # TODO: Store video_info in job
            filename=job.filename,
            video_path=video_path
        )
    except Exception as e:
        raise HTTPException(500, f"Failed to generate PDF: {str(e)}")
    
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="report_{job_id[:8]}.pdf"'
        }
    )
