"""
Grad-CAM Explainability Service: generates visual heatmaps and alpha-blended overlays
using native PyTorch gradient tracking on the final convolutional features.
"""

from typing import Tuple, Optional
from PIL import Image
import numpy as np
import torch

from src.gradcam import GradCAM, create_gradcam_overlay
from src.dataset import get_transforms
from src import CLASS_NAMES, CLASS_DESCRIPTIONS
from backend.app.services.model_service import model_manager
from backend.app.services.inference_service import pil_to_base64
from backend.app.schemas.prediction import GradCAMResult

val_transform = get_transforms("val")

def generate_gradcam_visualization(image: Image.Image, target_class_code: Optional[str] = None) -> GradCAMResult:
    """
    Computes Grad-CAM for the uploaded image using the loaded EfficientNetB0 backbone.
    Returns base64 strings for original image, heatmap, and overlay.
    """
    if not model_manager.is_loaded or model_manager.model is None:
        raise RuntimeError("AI model is not loaded. Cannot compute Grad-CAM.")

    rgb_image = image.convert("RGB")
    tensor = val_transform(rgb_image).unsqueeze(0).to(model_manager.device)

    # Resolve target class index
    target_idx = None
    if target_class_code and target_class_code in CLASS_NAMES:
        target_idx = CLASS_NAMES.index(target_class_code)

    # Initialize GradCAM engine
    gradcam = GradCAM(model_manager.model)
    try:
        heatmap_2d = gradcam.generate(tensor, target_class=target_idx)
    finally:
        gradcam.remove_hooks()

    # Determine predicted class
    if target_idx is None:
        with torch.no_grad():
            logits = model_manager.model(tensor)
            target_idx = int(torch.argmax(logits, dim=1).item())

    pred_code = CLASS_NAMES[target_idx]
    pred_name = CLASS_DESCRIPTIONS.get(pred_code, pred_code.upper())

    # Generate colored heatmap and overlay
    heatmap_uint8, overlay_uint8 = create_gradcam_overlay(
        original_pil_image=rgb_image,
        heatmap=heatmap_2d,
        alpha=0.45,
        colormap_name="jet"
    )

    heatmap_pil = Image.fromarray(heatmap_uint8)
    overlay_pil = Image.fromarray(overlay_uint8)

    original_b64 = pil_to_base64(rgb_image, format="JPEG")
    heatmap_b64 = pil_to_base64(heatmap_pil, format="JPEG")
    overlay_b64 = pil_to_base64(overlay_pil, format="JPEG")

    return GradCAMResult(
        predicted_code=pred_code,
        predicted_name=pred_name,
        original_b64=original_b64,
        heatmap_b64=heatmap_b64,
        overlay_b64=overlay_b64,
        target_layer="EfficientNetB0.features[-1]",
        description=(
            f"Grad-CAM visual activation map computed for class {pred_name} ({pred_code.upper()}). "
            "Warm colors (red/orange) indicate regions of morphological interest that strongly contributed to the model's prediction."
        )
    )
