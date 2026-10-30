"""
Model performance routes: serves real benchmark metrics from test split evaluation.
"""

from fastapi import APIRouter, HTTPException, status
from backend.app.schemas.performance import PerformanceMetrics
from backend.app.services.performance_service import get_model_performance

router = APIRouter()

@router.get("/performance", response_model=PerformanceMetrics)
async def fetch_performance():
    """
    Returns actual evaluation metrics on the held-out test split (1,494 images).
    """
    try:
        return get_model_performance()
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load performance metrics: {str(e)}"
        )
