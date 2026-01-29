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
    
    # Model settings
    MODEL_NAME: str = "google/owlv2-base-patch16-ensemble"
    CONFIDENCE_THRESHOLD: float = 0.15
    
    # Processing settings
    BATCH_SIZE: int = 8  # Frames per batch for inference
    SAMPLE_FPS: float = 5.0  # Analyze 5 frames per second (not all 30)
    MAX_UPLOAD_SIZE: int = 500 * 1024 * 1024  # 500MB
    
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
