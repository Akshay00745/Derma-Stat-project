"""
Singleton Model Service for loading and holding the trained EfficientNetB0 in memory.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional, Tuple

# Windows OpenMP protection
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import CLASS_NAMES, CLASS_DESCRIPTIONS, CLASS_COLORS
from src.model import build_model
from backend.app.core.config import settings

logger = logging.getLogger("dermastat.model_service")

class ModelManager:
    _instance: Optional["ModelManager"] = None
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.is_loaded = False
        self.confidence_threshold = settings.DEFAULT_CONFIDENCE_THRESHOLD
        self._load_calibration()
        self._load_model()
    
    @classmethod
    def get_instance(cls) -> "ModelManager":
        if cls._instance is None:
            cls._instance = ModelManager()
        return cls._instance

    def _load_calibration(self):
        if settings.CALIBRATION_PATH.exists():
            try:
                with open(settings.CALIBRATION_PATH, "r", encoding="utf-8") as f:
                    calib_data = json.load(f)
                    self.confidence_threshold = calib_data.get(
                        "optimal_threshold",
                        settings.DEFAULT_CONFIDENCE_THRESHOLD
                    )
                    logger.info(f"Loaded calibrated threshold: {self.confidence_threshold:.4f}")
            except Exception as e:
                logger.warning(f"Could not read calibration file: {e}")
                self.confidence_threshold = settings.DEFAULT_CONFIDENCE_THRESHOLD

    def _load_model(self):
        checkpoint_path = settings.CHECKPOINT_PATH
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found at: {checkpoint_path}")
            self.model = None
            self.is_loaded = False
            return
        
        try:
            logger.info(f"Loading checkpoint from: {checkpoint_path} onto {self.device}...")
            self.model = build_model(
                num_classes=len(CLASS_NAMES),
                pretrained=False,
                checkpoint_path=str(checkpoint_path),
                device=self.device
            )
            self.model.eval()
            self.is_loaded = True
            logger.info(f"Successfully loaded EfficientNetB0 on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            self.model = None
            self.is_loaded = False

model_manager = ModelManager.get_instance()
