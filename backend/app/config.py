"""
Application configuration
"""

from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    UPLOAD_DIR: str = "uploads"
    OUTPUT_DIR: str = "output"
    MODEL_CACHE_DIR: str = "models/optimized"
    
    # Model settings
    MODEL_NAME: str = "google/owlv2-base-patch16-ensemble"
    CONFIDENCE_THRESHOLD: float = 0.15
    
    # Processing settings
    BATCH_SIZE: int = 8  # Frames per batch for inference
    SAMPLE_FPS: float = 5.0  # Analyze 5 frames per second (not all 30)
    MAX_UPLOAD_SIZE: int = 500 * 1024 * 1024  # 500MB
    
    # Optimization settings
    ENABLE_OPTIMIZATION: bool = True  # Enable TensorRT/torch.compile optimization
    OPTIMIZATION_BACKEND: str = "auto"  # "auto", "tensorrt", "torch_compile", "eager"
    TENSORRT_PRECISION: str = "fp16"  # "fp32", "fp16", "int8"
    TORCH_COMPILE_MODE: str = "max-autotune"  # "default", "reduce-overhead", "max-autotune"
    WARMUP_ITERATIONS: int = 5  # Number of warmup iterations on startup
    
    # Default detection targets
    DEFAULT_QUERIES: list[str] = [
        "forklift",
        "AGV automated guided vehicle",
        "pallet",
        "person",
        "cardboard box",
        "conveyor belt"
    ]
    
    class Config:
        env_prefix = "IV_"


settings = Settings()
