"""
Validation-Calibrated Uncertainty Assessment Module.

Uses Maximum Softmax Probability (MSP) and validation-set empirical calibration:
- Calibrated strictly using the VALIDATION set (Test set is never used for tuning).
- Flags predictions below calibrated threshold as UNCERTAIN.
- Displays clear clinical explainers that uncertain predictions should not be trusted.
"""

import os
import json
from typing import Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_THRESHOLD = 0.65


def calibrate_uncertainty_threshold(
    model: torch.nn.Module,
    val_loader,
    device: torch.device,
    save_path: str = r"D:\HAM10000\checkpoints\uncertainty_calibration.json"
) -> Dict:
    """
    Calibrates confidence threshold using validation data only.
    Finds the threshold that optimizes separation between correct and incorrect predictions.
    """
    model.eval()
    max_probs = []
    correctness = []

    with torch.no_grad():
        for images, targets, _ in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            batch_max_probs, preds = torch.max(probs, dim=1)

            max_probs.extend(batch_max_probs.cpu().numpy())
            correctness.extend((preds == targets.to(device)).cpu().numpy())

    max_probs = np.array(max_probs)
    correctness = np.array(correctness)

    # Median / 75th percentile of incorrect validation predictions
    incorrect_probs = max_probs[~correctness]
    correct_probs = max_probs[correctness]

    if len(incorrect_probs) > 0:
        # Threshold set at 75th percentile of incorrect predictions to filter out most errors
        recommended_threshold = float(np.percentile(incorrect_probs, 75))
        # Bound between 0.50 and 0.85
        recommended_threshold = float(np.clip(recommended_threshold, 0.50, 0.85))
    else:
        recommended_threshold = DEFAULT_THRESHOLD

    calibration_data = {
        "calibrated_threshold": round(recommended_threshold, 4),
        "validation_samples": int(len(max_probs)),
        "correct_samples": int(np.sum(correctness)),
        "incorrect_samples": int(np.sum(~correctness)),
        "mean_confidence_correct": round(float(np.mean(correct_probs)), 4) if len(correct_probs) > 0 else 0.0,
        "mean_confidence_incorrect": round(float(np.mean(incorrect_probs)), 4) if len(incorrect_probs) > 0 else 0.0,
        "calibration_split": "validation_only"
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(calibration_data, f, indent=2)

    print(f"Calibrated Uncertainty Threshold: {recommended_threshold:.4f} (Saved to {save_path})")
    return calibration_data


def load_calibrated_threshold(
    calibration_path: str = r"D:\HAM10000\checkpoints\uncertainty_calibration.json"
) -> float:
    """Loads previously calibrated threshold or falls back to default."""
    if os.path.exists(calibration_path):
        try:
            with open(calibration_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return float(data.get("calibrated_threshold", DEFAULT_THRESHOLD))
        except Exception:
            return DEFAULT_THRESHOLD
    return DEFAULT_THRESHOLD


def assess_uncertainty(
    probabilities: np.ndarray,
    threshold: Optional[float] = None
) -> Dict:
    """
    Evaluates prediction confidence against the calibrated threshold.

    Args:
        probabilities: 1D array of 7 class softmax probabilities [0, 1].
        threshold: Optional custom threshold; if None, loads calibrated threshold.

    Returns:
        Structured evaluation dict.
    """
    if threshold is None:
        threshold = load_calibrated_threshold()

    top_idx = int(np.argmax(probabilities))
    top_prob = float(probabilities[top_idx])

    # Second highest for margin calculation
    sorted_probs = np.sort(probabilities)
    margin = float(sorted_probs[-1] - sorted_probs[-2]) if len(sorted_probs) > 1 else top_prob

    is_reliable = top_prob >= threshold

    if is_reliable:
        status = "RELIABLE"
        level = "High Confidence"
        message = (
            f"The model's top prediction confidence ({top_prob * 100:.1f}%) meets or exceeds "
            f"the calibrated validation threshold ({threshold * 100:.1f}%)."
        )
        recommendation = "Output represents a confident pattern match on benchmark dermoscopic features."
    else:
        status = "UNCERTAIN"
        level = "Low / Borderline Confidence"
        message = (
            f"WARNING: Model confidence ({top_prob * 100:.1f}%) falls below the calibrated validation "
            f"reliability threshold ({threshold * 100:.1f}%)."
        )
        recommendation = (
            "This prediction should be considered ambiguous or unreliable. Lesion features may overlap "
            "with multiple categories, or the image may contain non-standard artifacts."
        )

    return {
        "status": status,
        "is_reliable": is_reliable,
        "confidence_level": level,
        "top_confidence": round(top_prob, 4),
        "confidence_percentage": round(top_prob * 100, 2),
        "threshold_used": round(threshold, 4),
        "prediction_margin": round(margin, 4),
        "explanation": message,
        "recommendation": recommendation,
        "disclaimer": (
            "Educational ML prototype only. Confidence scores reflect statistical pattern recognition "
            "on the HAM10000 dataset, NOT clinical diagnostic certainty."
        )
    }
