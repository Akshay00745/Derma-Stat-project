"""
Dermastat AI Vision - FastAPI Application Entry Point.
Serves full-stack REST API and optionally mounts the built React frontend.
"""

import os
import sys
import logging
from contextlib import asynccontextmanager
from pathlib import Path

# OpenMP guard for Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Ensure HAM10000 root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from backend.app.core.config import settings
from backend.app.api.router import api_router
from backend.app.services.model_service import model_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("dermastat.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure model is ready
    logger.info("Initializing Dermastat AI Vision backend services...")
    if model_manager.is_loaded:
        logger.info(f"Model loaded and active on: {model_manager.device}")
    else:
        logger.warning("Model checkpoint could not be loaded on startup.")
    yield
    # Shutdown
    logger.info("Dermastat AI Vision backend shutting down.")


app = FastAPI(
    title="Dermastat AI Vision API",
    description="Full-stack Medical Deep Learning API for HAM10000 Skin Lesion Analysis & Grad-CAM Explainability",
    version="2.0.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled server error on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An internal server error occurred during request processing.",
            "error_type": exc.__class__.__name__
        }
    )

# Include API Router under /api
app.include_router(api_router, prefix=settings.API_V1_PREFIX)

# Serve static sample images
if settings.SAMPLES_DIR.exists():
    app.mount("/samples", StaticFiles(directory=str(settings.SAMPLES_DIR)), name="samples")

# Serve built frontend if dist exists
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
if FRONTEND_DIST.exists():
    logger.info(f"Mounting built frontend static files from: {FRONTEND_DIST}")
    assets_dir = FRONTEND_DIST / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        # Allow /api and /docs routes to pass through to FastAPI
        if full_path.startswith("api") or full_path.startswith("docs") or full_path.startswith("openapi.json"):
            return JSONResponse(status_code=404, content={"detail": "Not Found"})
        
        file_path = FRONTEND_DIST / full_path
        if full_path and file_path.exists() and file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(FRONTEND_DIST / "index.html"))
else:
    @app.get("/")
    async def root():
        return {
            "application": settings.PROJECT_NAME,
            "description": "AI-Powered Skin Lesion Classification & Explainability Web Application",
            "api_docs": "/docs",
            "health_check": f"{settings.API_V1_PREFIX}/health",
            "model_status": "Ready" if model_manager.is_loaded else "Degraded",
            "device": str(model_manager.device)
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
