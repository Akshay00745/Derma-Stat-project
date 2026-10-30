import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, r"D:\HAM10000")

from src import CLASS_NAMES, CLASS_DESCRIPTIONS
from src.model import build_model
from src.utils import validate_image_file, preprocess_pil_image_for_model, plot_probability_chart
from src.gradcam import GradCAM, create_gradcam_overlay
from src.uncertainty import load_calibrated_threshold, assess_uncertainty

def test_full_application_pipeline():
    print("=" * 65)
    print("PHASE 9: COMPREHENSIVE WEB APPLICATION FUNCTIONAL TEST")
    print("=" * 65)

    # 1. Model Loading
    print("\n[Test 1/6] Loading model checkpoint...")
    ckpt_path = r"D:\HAM10000\checkpoints\best_model.pth"
    assert os.path.exists(ckpt_path), "Checkpoint missing!"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=7, pretrained=False, checkpoint_path=ckpt_path, device=device)
    model.eval()
    print(f"  [PASS] Model loaded on {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # 2. Image Validation Testing
    print("\n[Test 2/6] Testing Image Validation and Error Handling...")
    # Test valid image
    sample_path = r"D:\HAM10000\data\raw\HAM10000_images_part_1\ISIC_0024306.jpg"
    with open(sample_path, "rb") as f:
        valid_bytes = f.read()
    is_valid, pil_img, msg = validate_image_file(valid_bytes)
    assert is_valid, f"Valid image failed: {msg}"
    print(f"  [PASS] Valid image check: {msg} ({pil_img.size})")

    # Test invalid bytes (corrupt)
    is_valid_bad, _, msg_bad = validate_image_file(b"NOT_AN_IMAGE_DATA")
    assert not is_valid_bad, "Corrupted image should be rejected!"
    print(f"  [PASS] Corrupted image rejection: {msg_bad}")

    # 3. Model Prediction Pipeline
    print("\n[Test 3/6] Testing 7-Class Prediction Pipeline...")
    input_tensor = preprocess_pil_image_for_model(pil_img, device=device)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    assert len(probs) == 7, "Must output 7 probabilities"
    assert abs(np.sum(probs) - 1.0) < 1e-4, "Probabilities must sum to 1.0"
    top_idx = int(np.argmax(probs))
    top_class = CLASS_NAMES[top_idx]
    top_prob = float(probs[top_idx])
    print(f"  [PASS] Prediction: {top_class.upper()} ({CLASS_DESCRIPTIONS[top_class]})")
    print(f"  [PASS] Top Confidence: {top_prob*100:.2f}%")
    print("  [PASS] 7-Class Confidence Distribution:")
    for c, p in zip(CLASS_NAMES, probs):
        print(f"         - {c.upper():5s}: {p*100:5.2f}%")

    # 4. Uncertainty Assessment Pipeline
    print("\n[Test 4/6] Testing Uncertainty Assessment Module...")
    thresh = load_calibrated_threshold()
    unc = assess_uncertainty(probs, threshold=thresh)
    print(f"  [PASS] Threshold in use: {unc['threshold_used']*100:.2f}%")
    print(f"  [PASS] Status: {unc['status']} ({unc['confidence_level']})")
    print(f"  [PASS] Explanation: {unc['explanation'][:80]}...")

    # 5. Grad-CAM Pipeline
    print("\n[Test 5/6] Testing Live Grad-CAM Generation & Overlay...")
    tensor_cam = input_tensor.clone().detach().requires_grad_(True)
    gradcam = GradCAM(model)
    cam = gradcam.generate(tensor_cam, target_class=top_idx)
    gradcam.remove_hooks()
    assert cam.shape == (224, 224), f"Cam shape mismatch: {cam.shape}"
    heatmap_colored, overlay = create_gradcam_overlay(pil_img, cam, alpha=0.45)
    assert overlay.shape == (pil_img.height, pil_img.width, 3), "Overlay dimension mismatch"
    print(f"  [PASS] Grad-CAM heatmap generated (min={cam.min():.2f}, max={cam.max():.2f})")
    print(f"  [PASS] Overlay generated with shape {overlay.shape}")

    # 6. Evaluation Artifacts Availability
    print("\n[Test 6/6] Verifying Performance Dashboard Artifacts...")
    results_json = r"D:\HAM10000\results\test_metrics.json"
    cm_png = r"D:\HAM10000\results\confusion_matrix.png"
    assert os.path.exists(results_json), "test_metrics.json missing!"
    assert os.path.exists(cm_png), "confusion_matrix.png missing!"
    print("  [PASS] Test metrics JSON and Confusion Matrix PNG confirmed on disk.")

    print("\n" + "=" * 65)
    print("ALL 6 WEB APPLICATION SUBSYSTEMS FUNCTIONING FLAWLESSLY!")
    print("=" * 65)

if __name__ == "__main__":
    test_full_application_pipeline()
