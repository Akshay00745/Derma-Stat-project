"""
Health check route.
"""

from fastapi import APIRouter
import torch

from backend.app.schemas.health import HealthResponse
from backend.app.services.model_service import model_manager
from backend.app.core.config import settings

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def check_health():
    return HealthResponse(
        status="healthy" if model_manager.is_loaded else "degraded",
        application=settings.PROJECT_NAME,
        version="2.0.0",
        model_loaded=model_manager.is_loaded,
        device=str(model_manager.device),
        cuda_available=torch.cuda.is_available()
    )
