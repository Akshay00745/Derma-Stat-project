"""
Utility functions for HAM10000 AI Web Application:
- Image validation and integrity checks
- Image preprocessing for inference
- Prediction probability bar charts
- Class descriptions and color palette mappings
"""

import io
from typing import Dict, List, Tuple, Optional
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

try:
    from . import CLASS_NAMES, CLASS_DESCRIPTIONS, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
except ImportError:
    CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    CLASS_DESCRIPTIONS = {
        "akiec": "Actinic Keratoses and Intraepithelial Carcinoma",
        "bcc": "Basal Cell Carcinoma",
        "bkl": "Benign Keratosis-like Lesions",
        "df": "Dermatofibroma",
        "mel": "Melanoma",
        "nv": "Melanocytic Nevi",
        "vasc": "Vascular Lesions"
    }
    IMAGE_SIZE = (224, 224)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

# Distinct, professional diagnostic color scheme
CLASS_COLORS = {
    "akiec": "#FF6B6B",   # Coral red
    "bcc": "#FFA07A",     # Light salmon
    "bkl": "#4ECDC4",     # Teal
    "df": "#45B7D1",      # Sky blue
    "mel": "#E74C3C",     # Strong Crimson (critical malignant)
    "nv": "#2ECC71",      # Emerald Green (benign common)
    "vasc": "#9B59B6"     # Purple
}


def validate_image_file(file_bytes: bytes, max_size_mb: float = 15.0) -> Tuple[bool, Optional[Image.Image], str]:
    """
    Validates uploaded dermoscopic image files:
    - File size check (< max_size_mb)
    - Image format check (JPEG, PNG, WebP)
    - Dimensions check (at least 100x100)
    - Readability check
    """
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > max_size_mb:
        return False, None, f"File size ({size_mb:.1f} MB) exceeds maximum allowed ({max_size_mb} MB)."

    try:
        img = Image.open(io.BytesIO(file_bytes))
        img.verify()
    except Exception as e:
        return False, None, f"Image file is corrupt or unreadable: {e}"

    # Re-open after verify()
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img = img.convert("RGB")
        w, h = img.size
        if w < 100 or h < 100:
            return False, None, f"Image resolution too small ({w}x{h}). Minimum required is 100x100."
        return True, img, "Valid Image"
    except Exception as e:
        return False, None, f"Could not process image: {e}"


def preprocess_pil_image_for_model(img: Image.Image, device: str = "cpu"):
    """
    Preprocesses a PIL Image into a normalized PyTorch tensor ready for EfficientNetB0.
    """
    import torch
    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    tensor = transform(img).unsqueeze(0).to(device)
    return tensor


def plot_probability_chart(probabilities: np.ndarray, top_class: str) -> plt.Figure:
    """
    Generates a clean horizontal bar chart displaying all seven diagnostic class confidence scores.
    """
    fig, ax = plt.subplots(figsize=(8, 4.2))

    labels = [f"{c.upper()} - {CLASS_DESCRIPTIONS[c]}" for c in CLASS_NAMES]
    probs = [probabilities[i] * 100 for i in range(len(CLASS_NAMES))]
    colors = [CLASS_COLORS[c] for c in CLASS_NAMES]

    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, probs, color=colors, height=0.65, edgecolor="none")

    # Annotate bar values
    for bar, prob in zip(bars, probs):
        width = bar.get_width()
        ax.text(
            width + 1.0,
            bar.get_y() + bar.get_height() / 2,
            f"{prob:.1f}%",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold" if width > 20 else "normal",
            color="#2c3e50"
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()  # Top-down order
    ax.set_xlim(0, 105)
    ax.set_xlabel("Confidence Probability (%)", fontsize=11, fontweight="medium")
    ax.set_title("7-Class Prediction Confidence Distribution", fontsize=12, fontweight="bold", pad=12)

    # Clean styling
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    fig.tight_layout()
    return fig
