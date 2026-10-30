"""
API Router aggregation.
"""

from fastapi import APIRouter
from backend.app.api.routes import (
    health,
    predictions,
    explainability,
    performance,
    samples,
    reports
)

api_router = APIRouter()

api_router.include_router(health.router, tags=["Health"])
api_router.include_router(predictions.router, tags=["Predictions"])
api_router.include_router(explainability.router, tags=["Explainability (Grad-CAM)"])
api_router.include_router(performance.router, tags=["Model Performance"])
api_router.include_router(samples.router, tags=["Sample Lesions"])
api_router.include_router(reports.router, tags=["Diagnostic Reports"])
