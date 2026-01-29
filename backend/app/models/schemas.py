"""
Pydantic models for API schemas
"""

from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from enum import Enum


class VideoStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Detection(BaseModel):
    """Single detection result"""
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]


class FrameDetection(BaseModel):
    """Detections for a single frame"""
    frame_number: int
    timestamp: float
    detections: List[Detection]
    counts: Dict[str, int]


class AnalysisSummary(BaseModel):
    """Summary of video analysis"""
    total_detections: int
    unique_classes: List[str]
    total_counts: Dict[str, int]
    max_simultaneous: Dict[str, int]
    frames_with_detections: int
    total_frames_analyzed: int


class VideoUploadResponse(BaseModel):
    """Response after video upload"""
    job_id: str
    filename: str
    status: VideoStatus
    message: str


class AnalysisJob(BaseModel):
    """Internal job tracking model"""
    job_id: str
    filename: str
    filepath: str
    status: VideoStatus = VideoStatus.PENDING
    progress: int = 0
    frames_processed: int = 0
    total_frames: int = 0
    config: Optional[Dict[str, Any]] = None
    summary: Optional[Dict[str, Any]] = None
    timeline: Optional[List[Dict[str, Any]]] = None
    output_path: Optional[str] = None
    
    class Config:
        use_enum_values = True
