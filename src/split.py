"""
Lesion-level Group Stratified Dataset Splitting Module.

Ensures strict separation across Train, Validation, and Test sets:
- Grouping on `lesion_id`: All images belonging to the same lesion ID reside
  exclusively in one split, preventing severe data leakage.
- Stratification on `dx`: Preserves class distribution proportions across splits.
- Test Split Isolation: The test split is reserved solely for final evaluation.
"""

import os
import csv
import json
import random
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

# Try relative import, fallback to absolute
try:
    from . import CLASS_NAMES, CLASS_TO_IDX
except ImportError:
    CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def build_image_lookup(raw_dir: str) -> Dict[str, str]:
    """Scans Part 1 and Part 2 folders to map image_id to exact image file path."""
    lookup = {}
    part1_dir = os.path.join(raw_dir, "HAM10000_images_part_1")
    part2_dir = os.path.join(raw_dir, "HAM10000_images_part_2")

    for folder in [part1_dir, part2_dir]:
        if not os.path.exists(folder):
            continue
        for fname in os.listdir(folder):
            if fname.lower().endswith(".jpg") and "(1)" not in fname:
                img_id = os.path.splitext(fname)[0]
                lookup[img_id] = os.path.join(folder, fname)
    return lookup


def create_stratified_group_splits(
    metadata_csv: str = r"D:\HAM10000\data\raw\HAM10000_metadata.csv",
    raw_dir: str = r"D:\HAM10000\data\raw",
    output_dir: str = r"D:\HAM10000\data\processed",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Dict:
    """
    Splits the HAM10000 dataset into Train, Val, and Test manifests
    using lesion-level grouping and stratified sampling.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, "Split ratios must sum to 1.0"

    print("=" * 65)
    print("PHASE 4: STRATIFIED LESION-LEVEL GROUP DATASET SPLITTING")
    print("=" * 65)

    # 1. Load image lookup
    print("\n[Step 1/5] Building image path index...")
    image_lookup = build_image_lookup(raw_dir)
    print(f"  Indexed {len(image_lookup)} image files on disk.")

    # 2. Read metadata CSV
    print("\n[Step 2/5] Reading metadata CSV records...")
    with open(metadata_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    print(f"  Total metadata records loaded: {len(rows)}")

    # Attach image_path and label_idx
    enriched_rows = []
    missing_count = 0
    for r in rows:
        img_id = r["image_id"]
        if img_id in image_lookup:
            row_copy = dict(r)
            row_copy["image_path"] = image_lookup[img_id]
            row_copy["label_idx"] = CLASS_TO_IDX[r["dx"]]
            enriched_rows.append(row_copy)
        else:
            missing_count += 1

    if missing_count > 0:
        raise FileNotFoundError(f"Found {missing_count} images in metadata missing from disk!")

    print(f"  Enriched {len(enriched_rows)} records with verified image paths and class indices.")

    # 3. Group rows by lesion_id
    print("\n[Step 3/5] Grouping records by unique lesion_id...")
    lesion_to_rows = defaultdict(list)
    for r in enriched_rows:
        lesion_to_rows[r["lesion_id"]].append(r)

    total_unique_lesions = len(lesion_to_rows)
    print(f"  Total unique lesions identified: {total_unique_lesions}")

    # Map each lesion to its diagnosis (dx)
    dx_to_lesions = defaultdict(list)
    for lid, rlist in lesion_to_rows.items():
        dx = rlist[0]["dx"]
        dx_to_lesions[dx].append(lid)

    # 4. Stratified lesion partitioning
    print(f"\n[Step 4/5] Partitioning lesions with seed={random_seed} (Train: {train_ratio*100:.0f}%, Val: {val_ratio*100:.0f}%, Test: {test_ratio*100:.0f}%)...")
    random.seed(random_seed)

    train_lesions = set()
    val_lesions = set()
    test_lesions = set()

    for dx, lids in sorted(dx_to_lesions.items()):
        # Deterministic shuffle
        random.shuffle(lids)
        n = len(lids)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        # Ensure at least 1 lesion in val and test for rare classes if count >= 3
        if n >= 3 and n_val == 0:
            n_val = 1
        
        train_lids = lids[:n_train]
        val_lids = lids[n_train:n_train + n_val]
        test_lids = lids[n_train + n_val:]

        train_lesions.update(train_lids)
        val_lesions.update(val_lids)
        test_lesions.update(test_lids)

    # Map lesions back to image rows
    train_rows = [r for r in enriched_rows if r["lesion_id"] in train_lesions]
    val_rows = [r for r in enriched_rows if r["lesion_id"] in val_lesions]
    test_rows = [r for r in enriched_rows if r["lesion_id"] in test_lesions]

    # 5. Strict Data Leakage Verification
    print("\n[Step 5/5] Running strict zero-leakage assertions...")
    leakage_train_val = train_lesions.intersection(val_lesions)
    leakage_train_test = train_lesions.intersection(test_lesions)
    leakage_val_test = val_lesions.intersection(test_lesions)

    assert len(leakage_train_val) == 0, f"DATA LEAKAGE DETECTED: {len(leakage_train_val)} lesions shared between Train and Val!"
    assert len(leakage_train_test) == 0, f"DATA LEAKAGE DETECTED: {len(leakage_train_test)} lesions shared between Train and Test!"
    assert len(leakage_val_test) == 0, f"DATA LEAKAGE DETECTED: {len(leakage_val_test)} lesions shared between Val and Test!"

    total_images_split = len(train_rows) + len(val_rows) + len(test_rows)
    assert total_images_split == len(enriched_rows), f"Image count mismatch! Expected {len(enriched_rows)}, got {total_images_split}"

    print("  Assertion PASS: Train-Val lesion overlap = 0")
    print("  Assertion PASS: Train-Test lesion overlap = 0")
    print("  Assertion PASS: Val-Test lesion overlap = 0")
    print(f"  Assertion PASS: Total images preserved = {total_images_split} / {len(enriched_rows)}")

    # Save manifest CSVs
    os.makedirs(output_dir, exist_ok=True)
    manifest_fields = ["lesion_id", "image_id", "dx", "label_idx", "dx_type", "age", "sex", "localization", "image_path"]

    train_csv_path = os.path.join(output_dir, "train.csv")
    val_csv_path = os.path.join(output_dir, "val.csv")
    test_csv_path = os.path.join(output_dir, "test.csv")

    for path, data_rows in [(train_csv_path, train_rows), (val_csv_path, val_rows), (test_csv_path, test_rows)]:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=manifest_fields)
            writer.writeheader()
            for r in data_rows:
                writer.writerow({k: r[k] for k in manifest_fields})
        print(f"  Saved manifest: {os.path.basename(path)} ({len(data_rows)} rows)")

    # Compute class distribution statistics
    c_train = Counter(r["dx"] for r in train_rows)
    c_val = Counter(r["dx"] for r in val_rows)
    c_test = Counter(r["dx"] for r in test_rows)

    stats = {
        "random_seed": random_seed,
        "total_records": len(enriched_rows),
        "total_unique_lesions": total_unique_lesions,
        "splits": {
            "train": {
                "images": len(train_rows),
                "images_percentage": round((len(train_rows) / len(enriched_rows)) * 100, 2),
                "lesions": len(train_lesions),
                "lesions_percentage": round((len(train_lesions) / total_unique_lesions) * 100, 2),
                "class_counts": dict(c_train)
            },
            "val": {
                "images": len(val_rows),
                "images_percentage": round((len(val_rows) / len(enriched_rows)) * 100, 2),
                "lesions": len(val_lesions),
                "lesions_percentage": round((len(val_lesions) / total_unique_lesions) * 100, 2),
                "class_counts": dict(c_val)
            },
            "test": {
                "images": len(test_rows),
                "images_percentage": round((len(test_rows) / len(enriched_rows)) * 100, 2),
                "lesions": len(test_lesions),
                "lesions_percentage": round((len(test_lesions) / total_unique_lesions) * 100, 2),
                "class_counts": dict(c_test)
            }
        },
        "leakage_verification": {
            "train_val_overlap": 0,
            "train_test_overlap": 0,
            "val_test_overlap": 0,
            "data_leakage_detected": False
        }
    }

    report_path = os.path.join(output_dir, "split_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\nSaved split report JSON to: {report_path}")
    print("=" * 65)

    return stats


if __name__ == "__main__":
    create_stratified_group_splits()
