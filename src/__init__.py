import os
# Prevent OpenMP runtime conflict on Windows with Intel MKL / Anaconda
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

"""
HAM10000 AI-Powered Skin Lesion Classification Package.
Provides modular components for dataset loading, lesion-level splitting,
EfficientNetB0 model training, Grad-CAM explainability, and uncertainty assessment.
"""

# The 7 official diagnostic categories in alphabetical order
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

CLASS_COLORS = {
    "akiec": "#EF4444",
    "bcc": "#F97316",
    "bkl": "#F59E0B",
    "df": "#10B981",
    "mel": "#8B5CF6",
    "nv": "#3B82F6",
    "vasc": "#EC4899"
}

CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}

# Standard image input parameters for EfficientNetB0
IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
