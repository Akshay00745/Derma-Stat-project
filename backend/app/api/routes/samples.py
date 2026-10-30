"""
Samples route: serves metadata and base64 previews for representative HAM10000 lesion categories.
"""

from typing import List
from pathlib import Path
from fastapi import APIRouter
from PIL import Image

from src import CLASS_NAMES, CLASS_DESCRIPTIONS
from backend.app.schemas.prediction import SampleItem
from backend.app.services.inference_service import pil_to_base64
from backend.app.core.config import settings

router = APIRouter()

SAMPLE_DESCRIPTIONS = {
    "akiec": "Actinic Keratoses and Intraepithelial Carcinoma (Pre-cancerous / Bowen's)",
    "bcc": "Basal Cell Carcinoma (Common malignant epithelial skin cancer)",
    "bkl": "Benign Keratosis-like Lesions (Seborrheic keratosis, solar lentigo)",
    "df": "Dermatofibroma (Benign fibrous histiocytoma)",
    "mel": "Melanoma (High-risk invasive cutaneous malignant neoplasm)",
    "nv": "Melanocytic Nevi (Common benign melanocytic mole)",
    "vasc": "Vascular Lesions (Angioma, angiokeratoma, pyogenic granuloma)"
}

@router.get("/samples", response_model=List[SampleItem])
async def list_sample_lesions():
    """
    Returns the list of 7 diagnostic category samples with thumbnails for UI rendering.
    """
    samples = []
    for code in CLASS_NAMES:
        img_path = settings.SAMPLES_DIR / f"{code}.jpg"
        if not img_path.exists():
            img_path = settings.SAMPLES_DIR / f"sample_{code}.jpg"
        
        thumb_b64 = ""
        if img_path.exists():
            try:
                img = Image.open(img_path)
                thumb_b64 = pil_to_base64(img, format="JPEG", quality=75)
            except Exception:
                thumb_b64 = ""

        samples.append(SampleItem(
            id=code,
            filename=f"{code}.jpg",
            code=code,
            name=CLASS_DESCRIPTIONS.get(code, code.upper()),
            description=SAMPLE_DESCRIPTIONS.get(code, ""),
            thumbnail_b64=thumb_b64
        ))
    return samples
