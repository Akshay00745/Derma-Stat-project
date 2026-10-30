"""
Comprehensive backend API verification script using FastAPI TestClient.
Tests all endpoints: health, sample prediction, Grad-CAM, performance, samples list, and report download.
"""

import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, "D:/HAM10000")

from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_all():
    print("=== STARTING BACKEND API TESTS ===")

    # 1. Health check
    print("\n[1/6] Testing GET /api/health...")
    resp = client.get("/api/health")
    assert resp.status_code == 200, f"Health check failed: {resp.text}"
    health_data = resp.json()
    print("  -> Status:", health_data["status"])
    print("  -> Model Loaded:", health_data["model_loaded"])
    print("  -> Device:", health_data["device"])
    print("  -> CUDA:", health_data["cuda_available"])
    assert health_data["model_loaded"] is True

    # 2. Samples list
    print("\n[2/6] Testing GET /api/samples...")
    resp = client.get("/api/samples")
    assert resp.status_code == 200
    samples = resp.json()
    print(f"  -> Found {len(samples)} representative sample categories")
    assert len(samples) == 7

    # 3. Model Performance metrics
    print("\n[3/6] Testing GET /api/performance...")
    resp = client.get("/api/performance")
    assert resp.status_code == 200
    perf = resp.json()
    print("  -> Test Accuracy:", perf["test_accuracy_pct"], "%")
    print("  -> Macro F1:", perf["macro_f1_pct"], "%")
    print("  -> Classes count:", len(perf["per_class"]))
    print("  -> Confusion Matrix rows:", len(perf["confusion_matrix"]))
    assert perf["test_accuracy_pct"] == 76.04
    assert perf["macro_f1_pct"] == 69.44

    # 4. Predict on sample image
    print("\n[4/6] Testing POST /api/predict/sample/mel...")
    resp = client.post("/api/predict/sample/mel")
    assert resp.status_code == 200, f"Sample prediction failed: {resp.text}"
    pred = resp.json()
    print("  -> Predicted Class:", pred["predicted_name"], f"({pred['predicted_code'].upper()})")
    print("  -> Confidence:", pred["confidence_percentage"], "%")
    print("  -> Uncertainty Status:", pred["uncertainty"]["status"], f"(Level: {pred['uncertainty']['level']})")
    print("  -> Top 3 Classes:")
    for prob in pred["all_probabilities"][:3]:
        print(f"     * {prob['name']}: {prob['percentage']}%")

    # 5. Grad-CAM on sample image
    print("\n[5/6] Testing POST /api/gradcam/sample/mel...")
    resp = client.post("/api/gradcam/sample/mel")
    assert resp.status_code == 200, f"Grad-CAM failed: {resp.text}"
    cam = resp.json()
    print("  -> Grad-CAM Class:", cam["predicted_name"])
    print("  -> Target Layer:", cam["target_layer"])
    print("  -> Original B64 prefix:", cam["original_b64"][:40])
    print("  -> Heatmap B64 prefix:", cam["heatmap_b64"][:40])
    print("  -> Overlay B64 prefix:", cam["overlay_b64"][:40])

    # 6. Report generation
    print("\n[6/6] Testing POST /api/reports/generate...")
    resp = client.post("/api/reports/generate", json=pred)
    assert resp.status_code == 200
    assert "attachment" in resp.headers.get("content-disposition", "")
    print("  -> Report downloaded successfully, bytes:", len(resp.content))

    print("\n=== ALL 6 BACKEND API TESTS PASSED PERFECTLY! ===")

if __name__ == "__main__":
    test_all()
