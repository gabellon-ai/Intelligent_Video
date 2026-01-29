"""
Intelligent Video - Backend API
FastAPI server for video upload, processing, and real-time streaming
"""

from fastapi import FastAPI, UploadFile, WebSocket, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import uuid
import os

from .config import settings
from .routers import videos, streams, analysis, exports, alerts, rtsp
from .services.detector import DetectorService
from .services.stream_service import init_stream_service, get_stream_service

# Global detector instance (loaded once at startup)
detector: DetectorService = None
stream_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML model at startup, cleanup on shutdown"""
    global detector, stream_service
    print("🚀 Loading detection model...")
    detector = DetectorService()
    await detector.load_model()
    print("✅ Model loaded, ready for inference")
    
    # Initialize stream service with detector
    stream_service = init_stream_service(detector)
    print("📹 Stream service initialized")
    
    yield
    
    # Cleanup streams on shutdown
    if stream_service:
        await stream_service.shutdown()
    print("👋 Shutting down...")


app = FastAPI(
    title="Intelligent Video API",
    description="Warehouse vision analysis - zero-shot object detection",
    version="0.1.0",
    lifespan=lifespan
)

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(videos.router, prefix="/api/videos", tags=["videos"])
app.include_router(streams.router, prefix="/api/streams", tags=["streams"])
app.include_router(rtsp.router, prefix="/api/rtsp", tags=["rtsp"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["analysis"])
app.include_router(exports.router, prefix="/api/videos", tags=["exports"])
app.include_router(alerts.router, prefix="/api/alerts", tags=["alerts"])

# Serve uploaded files
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")
app.mount("/output", StaticFiles(directory=settings.OUTPUT_DIR), name="output")


@app.get("/api/health")
async def health_check():
    active_streams = len(stream_service.list_streams()) if stream_service else 0
    return {
        "status": "healthy",
        "model_loaded": detector is not None and detector.is_loaded,
        "gpu_available": detector.gpu_available if detector else False,
        "active_streams": active_streams
    }


def get_detector() -> DetectorService:
    """Dependency to get detector instance"""
    return detector
