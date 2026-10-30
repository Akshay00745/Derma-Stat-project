"""
PyTorch Dataset and Image Preprocessing Module for HAM10000.

Configures:
- Image Resizing: 224x224 (EfficientNetB0 architecture standard)
- Image Normalization: ImageNet standard mean and standard deviation
- Training Augmentation: Horizontal/vertical flips, rotations, slight color jitter
- Validation/Test Transforms: Deterministic resize and normalization (NO augmentation)
- Class Weight Computation: Balanced inverse-frequency weights to address class imbalance
"""

import os
# Prevent OpenMP conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import csv
from typing import Dict, List, Optional, Tuple, Callable
from collections import Counter

# Try importing PIL and PyTorch, fallback gracefully for inspection environments
try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as transforms
except ImportError:
    torch = None
    Dataset = object
    DataLoader = None
    transforms = None

try:
    from . import CLASS_NAMES, CLASS_TO_IDX, IDX_TO_CLASS, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
except ImportError:
    CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}
    IMAGE_SIZE = (224, 224)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(split: str = "train") -> Optional[Callable]:
    """
    Returns torchvision transforms for the specified split.
    Augmentation is applied ONLY when split == 'train'.
    """
    if transforms is None:
        return None

    if split.lower() == "train":
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        # Validation, test, or inference: strictly deterministic, NO augmentation
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])


class SkinLesionDataset(Dataset):
    """
    PyTorch Dataset for HAM10000 Dermoscopic Images.
    Loads images on demand from disk and applies split-specific transformations.
    """

    def __init__(
        self,
        manifest_csv: str,
        split: str = "train",
        transform: Optional[Callable] = None
    ):
        self.manifest_csv = manifest_csv
        self.split = split
        self.transform = transform if transform is not None else get_transforms(split)

        if not os.path.exists(manifest_csv):
            raise FileNotFoundError(f"Manifest CSV not found: {manifest_csv}")

        self.samples = []
        with open(manifest_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                self.samples.append({
                    "image_id": r["image_id"],
                    "lesion_id": r["lesion_id"],
                    "dx": r["dx"],
                    "label_idx": int(r["label_idx"]),
                    "image_path": r["image_path"],
                    "dx_type": r.get("dx_type", ""),
                    "age": r.get("age", ""),
                    "sex": r.get("sex", ""),
                    "localization": r.get("localization", "")
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        img_path = sample["image_path"]

        if Image is None:
            raise RuntimeError("Pillow is required to load images. Install pillow.")

        try:
            # Ensure 3-channel RGB
            with Image.open(img_path) as raw_img:
                image = raw_img.convert("RGB")
        except Exception as e:
            raise IOError(f"Failed to open image at {img_path}: {e}")

        if self.transform is not None:
            image = self.transform(image)

        label = sample["label_idx"]
        return image, label, sample["image_id"]


def compute_class_weights(train_csv_path: str) -> Dict[str, float]:
    """
    Computes balanced class weights using inverse class frequency:
        weight_c = N_total / (N_classes * count_c)
    Returns both a dictionary of weights and a normalized weight tensor.
    """
    counts = Counter()
    with open(train_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            counts[r["dx"]] += 1

    total_samples = sum(counts.values())
    n_classes = len(CLASS_NAMES)

    weights_dict = {}
    weights_list = []

    for name in CLASS_NAMES:
        c = counts.get(name, 1)
        w = total_samples / (n_classes * c)
        weights_dict[name] = round(w, 4)
        weights_list.append(w)

    return {
        "class_counts": dict(counts),
        "weights_dict": weights_dict,
        "weights_list": weights_list
    }


def get_dataloader(
    manifest_csv: str,
    split: str = "train",
    batch_size: int = 32,
    shuffle: Optional[bool] = None,
    num_workers: int = 0
):
    """
    Creates and returns a PyTorch DataLoader for a given split manifest.
    """
    if DataLoader is None:
        raise RuntimeError("PyTorch is required for DataLoader.")

    if shuffle is None:
        shuffle = (split.lower() == "train")

    dataset = SkinLesionDataset(manifest_csv=manifest_csv, split=split)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available() if torch else False
    )
    return loader


if __name__ == "__main__":
    import json
    train_manifest = r"D:\HAM10000\data\processed\train.csv"
    if os.path.exists(train_manifest):
        w_info = compute_class_weights(train_manifest)
        print("Calculated Class Weights for Training (Inverse Frequency):")
        print(json.dumps(w_info, indent=2))
