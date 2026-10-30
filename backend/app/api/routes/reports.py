"""
Report generation route: constructs exportable diagnostic summary file.
"""

import json
from fastapi import APIRouter, Response
from backend.app.schemas.prediction import PredictionResult
from backend.app.services.report_service import build_diagnostic_report

router = APIRouter()

@router.post("/reports/generate")
async def generate_report_file(prediction: PredictionResult):
    """
    Builds a JSON report and delivers it as an attachment download.
    """
    report_data = build_diagnostic_report(prediction)
    json_bytes = json.dumps(report_data, indent=2).encode("utf-8")
    
    filename = f"Dermastat_Report_{prediction.predicted_code.upper()}_{prediction.confidence_percentage:.0f}pct.json"
    
    return Response(
        content=json_bytes,
        media_type="application/json",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
    )
