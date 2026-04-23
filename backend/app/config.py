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
    CONFIDENCE_THRESHOLD: float = 0.25  # Raised to reduce false positives

    # Processing settings (tuned for DGX Spark GB10)
    BATCH_SIZE: int = 8  # Frames per batch for GPU inference
    SAMPLE_FPS: float = 5.0  # Analyze 5 frames per second with GPU
    MAX_UPLOAD_SIZE: int = 12 * 1024 * 1024 * 1024  # 12GB

    # Optimization settings
    ENABLE_OPTIMIZATION: bool = True  # Enable TensorRT/torch.compile optimization
    OPTIMIZATION_BACKEND: str = "auto"  # "auto", "tensorrt", "torch_compile", "eager"
    TENSORRT_PRECISION: str = "fp16"  # "fp32", "fp16", "int8"
    TORCH_COMPILE_MODE: str = "max-autotune"  # "default", "reduce-overhead", "max-autotune"
    WARMUP_ITERATIONS: int = 5  # Number of warmup iterations on startup

    # Default detection targets
    DEFAULT_QUERIES: list[str] = [
        "person",
        "forklift",
        "wooden pallet",
    ]
    
    class Config:
        env_prefix = "IV_"


settings = Settings()
