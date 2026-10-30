"""
Prediction API routes: handles image upload, validation, and real model inference.
"""

import io
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from PIL import Image

from backend.app.schemas.prediction import PredictionResult
from backend.app.services.inference_service import run_inference
from backend.app.services.model_service import model_manager
from backend.app.core.config import settings

router = APIRouter()

@router.post("/predict/upload", response_model=PredictionResult)
async def predict_uploaded_image(file: UploadFile = File(...)):
    """
    Accepts an uploaded image file, validates its format and size,
    and runs genuine EfficientNetB0 inference.
    """
    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI model is not loaded. Cannot perform inference."
        )

    # Validate content type
    if file.content_type not in settings.ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: '{file.content_type}'. Please upload JPEG or PNG images."
        )

    # Read and validate file content
    contents = await file.read()
    if len(contents) > settings.MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {settings.MAX_UPLOAD_SIZE_BYTES // (1024*1024)}MB."
        )

    if len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty."
        )

    try:
        image = Image.open(io.BytesIO(contents))
        image.verify()  # Validate image integrity
        image = Image.open(io.BytesIO(contents))  # Reopen after verify
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Corrupted or invalid image file: {str(e)}"
        )

    try:
        result = run_inference(image)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(e)}"
        )


@router.post("/predict/sample/{sample_id}", response_model=PredictionResult)
async def predict_sample_image(sample_id: str):
    """
    Executes inference on one of the pre-cached representative HAM10000 sample images.
    """
    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI model is not loaded."
        )

    clean_id = sample_id.lower().replace(".jpg", "").replace(".png", "")
    sample_file = settings.SAMPLES_DIR / f"{clean_id}.jpg"
    
    if not sample_file.exists():
        # Check if sample_{clean_id}.jpg exists
        sample_file = settings.SAMPLES_DIR / f"sample_{clean_id}.jpg"

    if not sample_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sample lesion '{sample_id}' not found in {settings.SAMPLES_DIR}"
        )

    try:
        image = Image.open(sample_file)
        return run_inference(image)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process sample image: {str(e)}"
        )
