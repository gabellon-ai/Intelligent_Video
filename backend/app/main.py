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
from .routers import videos, streams, analysis
from .services.detector import DetectorService

# Global detector instance (loaded once at startup)
detector: DetectorService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML model at startup, cleanup on shutdown"""
    global detector
    print("🚀 Loading detection model...")
    detector = DetectorService()
    await detector.load_model()
    print("✅ Model loaded, ready for inference")
    yield
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
app.include_router(analysis.router, prefix="/api/analysis", tags=["analysis"])

# Serve uploaded files
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")
app.mount("/output", StaticFiles(directory=settings.OUTPUT_DIR), name="output")


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": detector is not None and detector.is_loaded,
        "gpu_available": detector.gpu_available if detector else False
    }


def get_detector() -> DetectorService:
    """Dependency to get detector instance"""
    return detector
