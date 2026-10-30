"""
Core application configuration and settings for Dermastat AI Vision API.
"""

import os
from pathlib import Path
from pydantic import BaseModel

# Project Root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

class Settings(BaseModel):
    PROJECT_NAME: str = "Dermastat AI Vision"
    API_V1_PREFIX: str = "/api"
    
    # Paths
    ROOT_DIR: Path = PROJECT_ROOT
    CHECKPOINT_PATH: Path = PROJECT_ROOT / "checkpoints" / "best_model.pth"
    CALIBRATION_PATH: Path = PROJECT_ROOT / "checkpoints" / "uncertainty_calibration.json"
    RESULTS_DIR: Path = PROJECT_ROOT / "results"
    SAMPLES_DIR: Path = PROJECT_ROOT / "samples"
    
    # Upload settings
    MAX_UPLOAD_SIZE_BYTES: int = 50 * 1024 * 1024  # 50 MB
    ALLOWED_IMAGE_TYPES: list[str] = ["image/jpeg", "image/png", "image/webp", "image/bmp"]
    
    # Model configuration
    NUM_CLASSES: int = 7
    IMAGE_SIZE: int = 224
    
    # Default fallback threshold if calibration file is missing
    DEFAULT_CONFIDENCE_THRESHOLD: float = 0.8315
    
    # CORS
    CORS_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "*"
    ]

settings = Settings()
