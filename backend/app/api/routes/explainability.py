"""
Grad-CAM Explainability routes.
"""

import io
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from PIL import Image

from backend.app.schemas.prediction import GradCAMResult
from backend.app.services.gradcam_service import generate_gradcam_visualization
from backend.app.services.model_service import model_manager
from backend.app.core.config import settings

router = APIRouter()

@router.post("/gradcam/upload", response_model=GradCAMResult)
async def explain_uploaded_image(
    file: UploadFile = File(...),
    target_class: Optional[str] = Form(None)
):
    """
    Computes Grad-CAM heatmaps for an uploaded dermoscopic image.
    """
    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI model is not loaded."
        )

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents))
        return generate_gradcam_visualization(image, target_class_code=target_class)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Grad-CAM generation failed: {str(e)}"
        )


@router.post("/gradcam/sample/{sample_id}", response_model=GradCAMResult)
async def explain_sample_image(
    sample_id: str,
    target_class: Optional[str] = None
):
    """
    Computes Grad-CAM for a pre-cached sample lesion.
    """
    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI model is not loaded."
        )

    clean_id = sample_id.lower().replace(".jpg", "").replace(".png", "")
    sample_file = settings.SAMPLES_DIR / f"{clean_id}.jpg"
    if not sample_file.exists():
        sample_file = settings.SAMPLES_DIR / f"sample_{clean_id}.jpg"

    if not sample_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sample lesion '{sample_id}' not found."
        )

    try:
        image = Image.open(sample_file)
        return generate_gradcam_visualization(image, target_class_code=target_class)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Grad-CAM generation failed: {str(e)}"
        )
