"""
Dermastat AI Vision - Backend ASGI Entry Point.
Enables running `uvicorn main:app --reload` directly from inside the `backend/` directory.
"""

import os
import sys
from pathlib import Path

# Prevent OpenMP runtime conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Resolve paths
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

# Ensure both project root and backend dir are in sys.path
for path_entry in [str(PROJECT_ROOT), str(BACKEND_DIR)]:
    if path_entry not in sys.path:
        sys.path.insert(0, path_entry)

# Import the FastAPI application instance
from backend.app.main import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
