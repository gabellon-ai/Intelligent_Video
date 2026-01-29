"""
Intelligent Video - Backend API
FastAPI server for video upload, processing, and real-time streaming

Supports TensorRT/torch.compile optimization for high-performance inference.
"""

from fastapi import FastAPI, UploadFile, WebSocket, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import uuid
import os
import logging

from .config import settings
from .routers import videos, streams, analysis, counting, exports, alerts, rtsp
from .services.detector import DetectorService
from .services.stream_service import init_stream_service, get_stream_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Global detector instance (loaded once at startup)
detector: DetectorService = None
stream_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML model at startup with optimization, cleanup on shutdown"""
    global detector, stream_service
    
    logger.info("🚀 Starting Intelligent Video API...")
    logger.info(f"   Model: {settings.MODEL_NAME}")
    logger.info(f"   Optimization enabled: {settings.ENABLE_OPTIMIZATION}")
    
    detector = DetectorService()
    await detector.load_model(optimize=settings.ENABLE_OPTIMIZATION)
    
    # Log optimization status
    opt_info = detector.optimization_info
    logger.info("✅ Model loaded successfully")
    logger.info(f"   Optimization backend: {opt_info.get('backend', 'none')}")
    logger.info(f"   Device: {detector.device}")
    logger.info(f"   GPU available: {detector.gpu_available}")
    
    if opt_info.get('speedup_estimate'):
        logger.info(f"   Estimated speedup: {opt_info['speedup_estimate']}x")
    
    # Run warmup
    if detector.device == "cuda" and settings.WARMUP_ITERATIONS > 0:
        logger.info(f"   Running {settings.WARMUP_ITERATIONS} warmup iterations...")
        detector.warmup(num_iterations=settings.WARMUP_ITERATIONS)
    
    # Initialize stream service with detector
    stream_service = init_stream_service(detector)
    logger.info("📹 Stream service initialized")
    
    logger.info("🎯 Ready for inference!")
    yield
    
    # Cleanup streams on shutdown
    if stream_service:
        await stream_service.shutdown()
    logger.info("👋 Shutting down...")


app = FastAPI(
    title="Intelligent Video API",
    description="Warehouse vision analysis - zero-shot object detection with TensorRT/torch.compile optimization",
    version="0.2.0",
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
app.include_router(counting.router, prefix="/api/counting", tags=["counting"])
app.include_router(exports.router, prefix="/api/videos", tags=["exports"])
app.include_router(alerts.router, prefix="/api/alerts", tags=["alerts"])

# Serve uploaded files
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")
app.mount("/output", StaticFiles(directory=settings.OUTPUT_DIR), name="output")


@app.get("/api/health")
async def health_check():
    """Health check endpoint with optimization status"""
    active_streams = len(stream_service.list_streams()) if stream_service else 0
    opt_info = detector.optimization_info if detector else {}
    
    return {
        "status": "healthy",
        "model_loaded": detector is not None and detector.is_loaded,
        "gpu_available": detector.gpu_available if detector else False,
        "active_streams": active_streams,
        "optimization": {
            "backend": opt_info.get("backend", "not_loaded"),
            "speedup_estimate": opt_info.get("speedup_estimate"),
            "cached": opt_info.get("cached", False)
        }
    }


@app.get("/api/optimization")
async def optimization_status():
    """Get detailed optimization status"""
    if not detector or not detector.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": settings.MODEL_NAME,
        "device": detector.device,
        "gpu_available": detector.gpu_available,
        "optimization": detector.optimization_info,
        "settings": {
            "enable_optimization": settings.ENABLE_OPTIMIZATION,
            "backend_preference": settings.OPTIMIZATION_BACKEND,
            "tensorrt_precision": settings.TENSORRT_PRECISION,
            "torch_compile_mode": settings.TORCH_COMPILE_MODE,
            "warmup_iterations": settings.WARMUP_ITERATIONS
        }
    }


def get_detector() -> DetectorService:
    """Dependency to get detector instance"""
    return detector
