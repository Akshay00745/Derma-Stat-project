"""
Report Service: generates a structured clinical diagnostic summary report for export.
"""

from datetime import datetime
from typing import Dict, Any

from backend.app.schemas.prediction import PredictionResult
from backend.app.core.config import settings

def build_diagnostic_report(prediction: PredictionResult, filename: str = "lesion.jpg") -> Dict[str, Any]:
    """
    Constructs a detailed analysis report dictionary ready for JSON download.
    """
    now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    
    return {
        "report_id": f"DVA-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
        "application": settings.PROJECT_NAME,
        "timestamp": now_str,
        "source_image": filename,
        "primary_prediction": {
            "category_code": prediction.predicted_code,
            "category_name": prediction.predicted_name,
            "confidence_score": prediction.confidence,
            "confidence_percentage": f"{prediction.confidence_percentage}%"
        },
        "uncertainty_assessment": {
            "status": prediction.uncertainty.status,
            "risk_level": prediction.uncertainty.level,
            "validation_threshold": f"{prediction.uncertainty.threshold * 100:.2f}%",
            "interpretation": prediction.uncertainty.message
        },
        "complete_class_distribution": [
            {
                "code": item.code.upper(),
                "name": item.name,
                "probability": round(item.probability, 4),
                "percentage": f"{item.percentage}%"
            }
            for item in prediction.all_probabilities
        ],
        "model_provenance": {
            "backbone": "EfficientNet-B0 (Transfer Learning)",
            "training_dataset": "HAM10000 (ViDIR Group, Medical University of Vienna)",
            "split_method": "Stratified Lesion-Level Isolation (0% Patient Leakage)",
            "calibration_threshold": f"{prediction.uncertainty.threshold * 100:.2f}%"
        },
        "medical_disclaimer": (
            "Dermastat AI Vision is an investigational and educational AI prototype developed "
            "for demonstration and clinical research. It is NOT a certified medical diagnostic device. "
            "This report must never substitute for biopsy, dermatoscopic evaluation, or clinical "
            "decision-making by a licensed physician or dermatologist."
        )
    }
