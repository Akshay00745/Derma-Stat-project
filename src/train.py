"""
Training Pipeline for HAM10000 7-Class Skin Lesion Classification.

Features:
- Transfer learning with EfficientNetB0 (ImageNet pretrained weights)
- Class-weighted CrossEntropyLoss to address severe class imbalance
- AdamW optimizer with weight decay
- Learning rate scheduling with ReduceLROnPlateau / CosineAnnealing
- Validation monitoring after each epoch (Accuracy, Loss, Macro-F1)
- Best checkpoint saving based strictly on Validation Macro-F1
- Early stopping mechanism
- Mixed precision training (torch.amp) for GPU efficiency and speed
- Training history logging and visual curve plotting
"""

import os
import sys

# Prevent OpenMP runtime conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import json
import argparse
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import autocast, GradScaler
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt

# Ensure src package can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import CLASS_NAMES, CLASS_TO_IDX
from src.dataset import get_dataloader, compute_class_weights
from src.model import build_model


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device
) -> Tuple[float, float, float]:
    """Runs a single training epoch across the training dataset."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for images, targets, _ in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Mixed precision forward pass
        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, targets)

        # Backward pass with scaled gradients
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    epoch_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    return epoch_loss, epoch_acc, epoch_f1


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, float]:
    """Runs a validation evaluation epoch across the validation dataset."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for images, targets, _ in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, targets)

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    epoch_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    return epoch_loss, epoch_acc, epoch_f1


def plot_and_save_curves(history: Dict, save_path: str):
    """Generates and saves professional training and validation curves."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss curve
    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], "r--s", label="Val Loss", linewidth=2)
    axes[0].set_title("Cross-Entropy Loss Progression", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Loss", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(fontsize=11)

    # Accuracy curve
    axes[1].plot(epochs, history["train_acc"], "b-o", label="Train Acc", linewidth=2)
    axes[1].plot(epochs, history["val_acc"], "r--s", label="Val Acc", linewidth=2)
    axes[1].set_title("Accuracy Progression", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Accuracy", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(fontsize=11)

    # Macro-F1 curve
    axes[2].plot(epochs, history["train_f1"], "b-o", label="Train Macro-F1", linewidth=2)
    axes[2].plot(epochs, history["val_f1"], "g--^", label="Val Macro-F1", linewidth=2)
    axes[2].set_title("Macro-F1 Score (Validation Criterion)", fontsize=13, fontweight="bold")
    axes[2].set_xlabel("Epoch", fontsize=11)
    axes[2].set_ylabel("Macro-F1", fontsize=11)
    axes[2].grid(True, linestyle="--", alpha=0.6)
    axes[2].legend(fontsize=11)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved training curve plots to: {save_path}")


def run_training(
    train_csv: str = r"D:\HAM10000\data\processed\train.csv",
    val_csv: str = r"D:\HAM10000\data\processed\val.csv",
    checkpoints_dir: str = r"D:\HAM10000\checkpoints",
    metrics_dir: str = r"D:\HAM10000\metrics",
    num_epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    patience: int = 5,
    num_workers: int = 2
) -> Dict:
    """Executes the complete transfer learning training pipeline."""
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    print("=" * 70)
    print("PHASE 5: EFFICIENTNET-B0 MODEL DEVELOPMENT AND TRAINING")
    print("=" * 70)

    # Hardware detection
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"Hardware Compute: GPU Detected -> {gpu_name} ({total_vram:.2f} GB VRAM)")
    else:
        device = torch.device("cpu")
        print("Hardware Compute: CUDA not available. Running on CPU.")
        batch_size = min(batch_size, 16)
        num_workers = 0

    print(f"Training Parameters: Batch Size={batch_size}, LR={lr}, Max Epochs={num_epochs}, Early Stopping Patience={patience}")

    # Compute balanced class weights for loss weighting
    weights_info = compute_class_weights(train_csv)
    weight_tensor = torch.tensor(weights_info["weights_list"], dtype=torch.float32).to(device)
    print("\nClass Balancing Weights Applied to CrossEntropyLoss:")
    for cls_name, w in weights_info["weights_dict"].items():
        print(f"  - {cls_name.upper():5s}: {w:.4f}")

    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    # DataLoaders
    print("\nInitializing PyTorch DataLoaders...")
    train_loader = get_dataloader(train_csv, split="train", batch_size=batch_size, num_workers=num_workers)
    val_loader = get_dataloader(val_csv, split="val", batch_size=batch_size, num_workers=num_workers)
    print(f"  Train samples: {len(train_loader.dataset)} across {len(train_loader)} batches")
    print(f"  Val samples:   {len(val_loader.dataset)} across {len(val_loader)} batches")

    # Build model
    print("\nInstantiating EfficientNetB0 with ImageNet pretrained weights...")
    model = build_model(num_classes=len(CLASS_NAMES), pretrained=True, dropout=0.3, device=device)

    # Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2, verbose=True)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    # Tracking metrics
    history = {
        "train_loss": [],
        "train_acc": [],
        "train_f1": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "lr": []
    }

    best_val_f1 = -1.0
    best_epoch = -1
    best_checkpoint_path = os.path.join(checkpoints_dir, "best_model.pth")
    early_stop_counter = 0

    total_start_time = time.time()
    print("\nStarting Training Loop...")
    print("-" * 75)
    print(f"{'Epoch':<6} | {'Train Loss':<10} | {'Train Acc':<10} | {'Val Loss':<10} | {'Val Acc':<10} | {'Val F1 (Macro)':<14} | {'LR':<9}")
    print("-" * 75)

    for epoch in range(1, num_epochs + 1):
        t_epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        # Train & Evaluate
        train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc, val_f1 = evaluate_epoch(model, val_loader, criterion, device)

        # Step scheduler on validation Macro-F1
        scheduler.step(val_f1)

        # Log history
        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["train_f1"].append(round(train_f1, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 4))
        history["val_f1"].append(round(val_f1, 4))
        history["lr"].append(current_lr)

        epoch_time = time.time() - t_epoch_start
        is_best = val_f1 > best_val_f1

        print(f"{epoch:<6} | {train_loss:<10.4f} | {train_acc*100:<9.2f}% | {val_loss:<10.4f} | {val_acc*100:<9.2f}% | {val_f1*100:<13.2f}% | {current_lr:<9.1e} ({epoch_time:.1f}s){' *BEST*' if is_best else ''}")

        # Checkpoint if best validation Macro-F1
        if is_best:
            best_val_f1 = val_f1
            best_epoch = epoch
            early_stop_counter = 0

            checkpoint_data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_macro_f1": val_f1,
                "val_accuracy": val_acc,
                "val_loss": val_loss,
                "class_names": CLASS_NAMES,
                "class_weights": weights_info["weights_dict"],
                "architecture": "efficientnet_b0"
            }
            torch.save(checkpoint_data, best_checkpoint_path)
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"\n[Early Stopping Triggered] No improvement in Validation Macro-F1 for {patience} consecutive epochs.")
                break

    total_duration = time.time() - total_start_time

    # Save training history JSON
    history_path = os.path.join(checkpoints_dir, "training_history.json")
    summary = {
        "best_epoch": best_epoch,
        "best_val_macro_f1": round(best_val_f1, 4),
        "total_training_duration_seconds": round(total_duration, 1),
        "total_epochs_trained": len(history["train_loss"]),
        "history": history
    }
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved training history to: {history_path}")

    # Plot curves
    plot_path = os.path.join(metrics_dir, "training_curves.png")
    plot_and_save_curves(history, plot_path)

    # Calibrate uncertainty threshold on validation set only
    print("\nCalibrating uncertainty threshold on validation dataset...")
    try:
        from src.uncertainty import calibrate_uncertainty_threshold
        eval_best_model = build_model(num_classes=len(CLASS_NAMES), pretrained=False, checkpoint_path=best_checkpoint_path, device=device)
        calibrate_uncertainty_threshold(eval_best_model, val_loader, device)
    except Exception as e:
        print(f"Warning during uncertainty calibration: {e}")

    print("=" * 70)
    print(f"TRAINING COMPLETE in {total_duration / 60:.1f} minutes")
    print(f"Best Validation Macro-F1: {best_val_f1 * 100:.2f}% (Achieved at Epoch {best_epoch})")
    print(f"Best Checkpoint: {best_checkpoint_path}")
    print("=" * 70)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EfficientNetB0 on HAM10000")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of dataloader workers")
    args = parser.parse_args()

    run_training(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        num_workers=args.num_workers
    )
