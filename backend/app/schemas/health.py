"""
Health check schema models.
"""

from pydantic import BaseModel

class HealthResponse(BaseModel):
    status: str
    application: str
    version: str
    model_loaded: bool
    device: str
    cuda_available: bool
