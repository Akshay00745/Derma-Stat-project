"""
Prediction, uncertainty, and Grad-CAM schemas.
"""

from typing import List, Optional
from pydantic import BaseModel

class ClassConfidence(BaseModel):
    code: str
    name: str
    probability: float
    percentage: float
    color: str

class UncertaintyInfo(BaseModel):
    threshold: float
    confidence: float
    is_uncertain: bool
    status: str
    level: str
    message: str

class PredictionResult(BaseModel):
    predicted_code: str
    predicted_name: str
    confidence: float
    confidence_percentage: float
    is_confident: bool
    uncertainty: UncertaintyInfo
    all_probabilities: List[ClassConfidence]
    image_preview_b64: Optional[str] = None
    original_dimensions: Optional[dict] = None

class GradCAMResult(BaseModel):
    predicted_code: str
    predicted_name: str
    original_b64: str
    heatmap_b64: str
    overlay_b64: str
    target_layer: str
    description: str

class SampleItem(BaseModel):
    id: str
    filename: str
    code: str
    name: str
    description: str
    thumbnail_b64: str
