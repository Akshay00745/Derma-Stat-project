"""
Model Evaluation Pipeline for HAM10000 7-Class Skin Lesion Classification.

Evaluates the trained EfficientNetB0 checkpoint (checkpoints/best_model.pth)
exclusively on the reserved test dataset (data/processed/test.csv):
- Test Accuracy
- Macro and Weighted Precision
- Macro and Weighted Recall
- Macro-F1 Score and Weighted-F1 Score
- Per-class Classification Report for all 7 classes (AKIEC, BCC, BKL, DF, MEL, NV, VASC)
- Raw Confusion Matrix Plot (PNG)
- Normalized Confusion Matrix Plot (PNG)
- Numerical Metrics saved in JSON and CSV formats inside results/
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import json
import csv
import argparse
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Ensure src package can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import CLASS_NAMES, CLASS_DESCRIPTIONS
from src.dataset import get_dataloader
from src.model import build_model


def plot_raw_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: str):
    """Plots and saves the raw count confusion matrix."""
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    labels = [c.upper() for c in class_names]
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        title="HAM10000 7-Class Confusion Matrix (Raw Test Counts)",
        ylabel="True Ground Truth Diagnosis",
        xlabel="Predicted Diagnosis"
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontweight="bold"
            )

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved raw confusion matrix to: {save_path}")


def plot_normalized_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: str):
    """Plots and saves the normalized (percentage per true class) confusion matrix."""
    # Normalize by true row sum
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0.0, vmax=1.0)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    labels = [c.upper() for c in class_names]
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        title="HAM10000 7-Class Normalized Confusion Matrix (Recall %)",
        ylabel="True Ground Truth Diagnosis",
        xlabel="Predicted Diagnosis"
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm_norm[i, j]
            ax.text(
                j, i, f"{val * 100:.1f}%",
                ha="center", va="center",
                color="white" if val > thresh else "black",
                fontweight="bold"
            )

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved normalized confusion matrix to: {save_path}")


def run_evaluation(
    checkpoint_path: str = r"D:\HAM10000\checkpoints\best_model.pth",
    test_csv: str = r"D:\HAM10000\data\processed\test.csv",
    results_dir: str = r"D:\HAM10000\results",
    metrics_dir: str = r"D:\HAM10000\metrics",
    batch_size: int = 32
) -> Dict:
    """Executes final evaluation on the reserved test split."""
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("PHASE 6: FINAL MODEL EVALUATION ON RESERVED TEST SET")
    print("=" * 70)
    print(f"Compute Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"Model Checkpoint: {checkpoint_path}")
    print(f"Test Manifest:    {test_csv}")

    # 1. Load model checkpoint
    print("\n[Step 1/5] Loading trained EfficientNetB0 checkpoint...")
    model = build_model(
        num_classes=len(CLASS_NAMES),
        pretrained=False,
        checkpoint_path=checkpoint_path,
        device=device
    )
    model.eval()

    # 2. Load test dataset
    print("\n[Step 2/5] Initializing Test DataLoader...")
    test_loader = get_dataloader(test_csv, split="test", batch_size=batch_size, shuffle=False)
    print(f"  Test samples: {len(test_loader.dataset)} images across {len(test_loader)} batches")

    # 3. Inference pass
    print("\n[Step 3/5] Running single-pass evaluation across all test samples...")
    all_targets = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, targets, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # 4. Calculate metrics
    print("\n[Step 4/5] Computing comprehensive evaluation metrics...")
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)
    macro_prec = precision_score(all_targets, all_preds, average="macro", zero_division=0)
    weighted_prec = precision_score(all_targets, all_preds, average="weighted", zero_division=0)
    macro_rec = recall_score(all_targets, all_preds, average="macro", zero_division=0)
    weighted_rec = recall_score(all_targets, all_preds, average="weighted", zero_division=0)

    # Per-class report
    clf_report_dict = classification_report(
        all_targets,
        all_preds,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    clf_report_text = classification_report(
        all_targets,
        all_preds,
        target_names=CLASS_NAMES,
        zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds, labels=range(len(CLASS_NAMES)))

    # 5. Save Artifacts
    print("\n[Step 5/5] Saving evaluation artifacts to results/ and metrics/...")

    # Plot Confusion Matrices
    cm_raw_path = os.path.join(results_dir, "confusion_matrix.png")
    plot_raw_confusion_matrix(cm, CLASS_NAMES, cm_raw_path)
    # Also save to metrics/ for web app
    plot_raw_confusion_matrix(cm, CLASS_NAMES, os.path.join(metrics_dir, "confusion_matrix.png"))

    cm_norm_path = os.path.join(results_dir, "confusion_matrix_normalized.png")
    plot_normalized_confusion_matrix(cm, CLASS_NAMES, cm_norm_path)

    # Per-class table to CSV
    csv_rows = []
    for cls_name in CLASS_NAMES:
        c_dict = clf_report_dict[cls_name]
        csv_rows.append({
            "class_code": cls_name.upper(),
            "diagnostic_category": CLASS_DESCRIPTIONS[cls_name],
            "precision": round(c_dict["precision"], 4),
            "recall": round(c_dict["recall"], 4),
            "f1_score": round(c_dict["f1-score"], 4),
            "support": int(c_dict["support"])
        })

    report_csv_path = os.path.join(results_dir, "classification_report.csv")
    with open(report_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class_code", "diagnostic_category", "precision", "recall", "f1_score", "support"])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Saved classification report CSV to: {report_csv_path}")

    # Structured JSON
    results_json = {
        "test_dataset": "HAM10000 (reserved test split)",
        "total_test_samples": len(all_targets),
        "overall_metrics": {
            "test_accuracy": round(acc, 4),
            "test_accuracy_percentage": round(acc * 100, 2),
            "macro_f1": round(macro_f1, 4),
            "macro_f1_percentage": round(macro_f1 * 100, 2),
            "weighted_f1": round(weighted_f1, 4),
            "weighted_f1_percentage": round(weighted_f1 * 100, 2),
            "macro_precision": round(macro_prec, 4),
            "macro_precision_percentage": round(macro_prec * 100, 2),
            "weighted_precision": round(weighted_prec, 4),
            "weighted_precision_percentage": round(weighted_prec * 100, 2),
            "macro_recall": round(macro_rec, 4),
            "macro_recall_percentage": round(macro_rec * 100, 2),
            "weighted_recall": round(weighted_rec, 4),
            "weighted_recall_percentage": round(weighted_rec * 100, 2)
        },
        "per_class_metrics": {
            cls_name: {
                "name": CLASS_DESCRIPTIONS[cls_name],
                "precision": round(clf_report_dict[cls_name]["precision"], 4),
                "recall": round(clf_report_dict[cls_name]["recall"], 4),
                "f1_score": round(clf_report_dict[cls_name]["f1-score"], 4),
                "support": int(clf_report_dict[cls_name]["support"])
            }
            for cls_name in CLASS_NAMES
        },
        "confusion_matrix": cm.tolist()
    }

    results_json_path = os.path.join(results_dir, "test_metrics.json")
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2)
    print(f"Saved test metrics JSON to: {results_json_path}")

    # Mirror for Web App
    with open(os.path.join(metrics_dir, "evaluation_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results_json["overall_metrics"] | {"per_class_metrics": results_json["per_class_metrics"], "overall_accuracy": round(acc, 4), "confusion_matrix": cm.tolist()}, f, indent=2)

    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Test Accuracy:       {acc * 100:.2f}%")
    print(f"Macro-F1 Score:      {macro_f1 * 100:.2f}%")
    print(f"Weighted-F1 Score:   {weighted_f1 * 100:.2f}%")
    print(f"Macro Precision:     {macro_prec * 100:.2f}%")
    print(f"Weighted Precision:  {weighted_prec * 100:.2f}%")
    print(f"Macro Recall:        {macro_rec * 100:.2f}%")
    print(f"Weighted Recall:     {weighted_rec * 100:.2f}%")
    print("\nDetailed Per-Class Report:")
    print(clf_report_text)
    print("=" * 70)

    return results_json


if __name__ == "__main__":
    run_evaluation()
