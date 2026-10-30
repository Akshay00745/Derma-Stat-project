"""
Inference Service: handles image preprocessing, model execution, probability computation,
and validation-calibrated uncertainty evaluation.
"""

import io
import base64
from typing import Tuple, Dict, Any
from PIL import Image
import torch
import torch.nn.functional as F

from src import CLASS_NAMES, CLASS_DESCRIPTIONS, CLASS_COLORS
from src.dataset import get_transforms
from backend.app.services.model_service import model_manager
from backend.app.schemas.prediction import (
    ClassConfidence,
    UncertaintyInfo,
    PredictionResult
)

# Get validation/inference transformation (Resize 224x224, Normalize with ImageNet stats)
val_transform = get_transforms("val")

def pil_to_base64(img: Image.Image, format: str = "JPEG", quality: int = 85) -> str:
    """Converts a PIL Image to a base64 data URI string."""
    buffered = io.BytesIO()
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.save(buffered, format=format, quality=quality)
    b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{b64_str}"

def run_inference(image: Image.Image) -> PredictionResult:
    """
    Executes deep learning inference on a given PIL Image.
    Returns the top predicted class, confidence, calibrated uncertainty,
    and the full 7-class probability distribution.
    """
    if not model_manager.is_loaded or model_manager.model is None:
        raise RuntimeError("AI model is not loaded. Please verify checkpoints/best_model.pth.")
    
    # Store original dimensions
    orig_w, orig_h = image.size
    
    # Preprocess image
    rgb_img = image.convert("RGB")
    img_tensor = val_transform(rgb_img).unsqueeze(0).to(model_manager.device)
    
    # Run model forward pass
    with torch.no_grad():
        logits = model_manager.model(img_tensor)
        probabilities = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    
    # Compute top prediction
    top_idx = int(probabilities.argmax())
    top_code = CLASS_NAMES[top_idx]
    top_name = CLASS_DESCRIPTIONS.get(top_code, top_code.upper())
    top_confidence = float(probabilities[top_idx])
    
    # Uncertainty calibration assessment
    threshold = model_manager.confidence_threshold
    is_uncertain = top_confidence < threshold
    
    if is_uncertain:
        uncertainty_status = "Uncertain"
        uncertainty_level = "High"
        uncertainty_message = (
            f"Confidence ({top_confidence * 100:.1f}%) is below the validation-calibrated "
            f"safety threshold ({threshold * 100:.1f}%). Clinical review is strongly recommended."
        )
    else:
        uncertainty_status = "Reliable"
        uncertainty_level = "Low"
        uncertainty_message = (
            f"Confidence ({top_confidence * 100:.1f}%) meets the validation-calibrated "
            f"safety criteria (≥ {threshold * 100:.1f}%). The prediction has high statistical confidence."
        )
    
    uncertainty_info = UncertaintyInfo(
        threshold=threshold,
        confidence=top_confidence,
        is_uncertain=is_uncertain,
        status=uncertainty_status,
        level=uncertainty_level,
        message=uncertainty_message
    )
    
    # Build complete 7-class distribution sorted by probability descending
    all_probs = []
    for idx, code in enumerate(CLASS_NAMES):
        prob = float(probabilities[idx])
        all_probs.append(ClassConfidence(
            code=code,
            name=CLASS_DESCRIPTIONS.get(code, code.upper()),
            probability=prob,
            percentage=round(prob * 100.0, 2),
            color=CLASS_COLORS.get(code, "#3B82F6")
        ))
    
    all_probs.sort(key=lambda x: x.probability, reverse=True)
    
    # Base64 thumbnail for preview
    preview_b64 = pil_to_base64(rgb_img, format="JPEG", quality=85)
    
    return PredictionResult(
        predicted_code=top_code,
        predicted_name=top_name,
        confidence=top_confidence,
        confidence_percentage=round(top_confidence * 100.0, 2),
        is_confident=not is_uncertain,
        uncertainty=uncertainty_info,
        all_probabilities=all_probs,
        image_preview_b64=preview_b64,
        original_dimensions={"width": orig_w, "height": orig_h}
    )
