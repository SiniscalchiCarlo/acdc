from __future__ import annotations

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from monai.transforms import AsDiscrete, Compose, EnsureType

# Allow the script to be executed from the repository root without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import training_config as cfg
from training import get_device, get_model
from src.load_data_2D import (
    build_preprocessed_2d_list,
    load_preprocessed_2d_manifest,
    split_by_patient,
    build_loaders,
)
from src.transforms_2D import build_preprocessed_val_transform
from monai.data import pad_list_data_collate

# -----------------------------
# Settings
# -----------------------------
# Number of validation images to visualize and save.
NUM_IMAGES = 5

# Path to the trained model checkpoint to load.
MODEL_OUTPUT = Path("artifacts") / "models" / "ATTENUNET_2.5D.pt"

# Directory where segmentation visualizations will be saved.
OUTPUT_DIR = Path("artifacts") / "segmentation_visuals" / "baseline_attenunet_2.5D"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Class definitions
# -----------------------------
# Colours used for contour overlays per class.
CLASS_COLORS = {
    1: "red",    # RV
    2: "green",  # MYO
    3: "blue",   # LV
}

# Class names for the legend.
CLASS_NAMES = {1: "RV", 2: "MYO", 3: "LV"}


def load_trained_model(model_path: Path, device: torch.device) -> torch.nn.Module:
    """Load a trained model checkpoint from disk and set it to evaluation mode."""
    model = get_model(cfg).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def visualize_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    preds: torch.Tensor,
    batch_idx: int,
    output_dir: Path,
) -> None:
    """Save side-by-side visualizations of input, ground truth, and prediction for a batch."""
    from matplotlib.lines import Line2D

    batch_size = images.shape[0]
    for i in range(batch_size):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        axes[0].imshow(images[i, 0].cpu().numpy(), cmap="gray")
        axes[0].set_title("Input MRI")
        axes[0].axis("off")

        # Ground truth contours overlaid on the image
        axes[1].imshow(images[i, 0].cpu().numpy(), cmap="gray")
        gt = labels[i, 0].cpu().numpy().squeeze()
        for class_idx, color in CLASS_COLORS.items():
            mask = gt == class_idx
            if mask.any():
                axes[1].contour(mask, colors=color, levels=[0.5], linewidths=1.5)
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")

        # Predicted contours overlaid on the image
        axes[2].imshow(images[i, 0].cpu().numpy(), cmap="gray")
        pred = preds[i].cpu().numpy().squeeze()
        for class_idx, color in CLASS_COLORS.items():
            mask = pred == class_idx
            if mask.any():
                axes[2].contour(mask, colors=color, linewidths=1.5)

        # Add legend for class colours
        legend = [Line2D([0], [0], color=c, label=CLASS_NAMES[j])
                  for j, c in CLASS_COLORS.items()]
        axes[2].legend(handles=legend, loc="upper right", fontsize=8)
        axes[2].set_title("Prediction")
        axes[2].axis("off")

        plt.tight_layout()
        save_path = output_dir / f"seg_batch{batch_idx}_img{i}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")


def main() -> None:
    """Load a trained model and save segmentation visualizations for validation slices."""
    device = get_device()
    model = load_trained_model(MODEL_OUTPUT, device)

    # Load preprocessed validation data
    manifest = load_preprocessed_2d_manifest(cfg.PREPROCESSED_ROOT)
    items = build_preprocessed_2d_list(cfg.PREPROCESSED_ROOT)
    _, val_items = split_by_patient(items, val_size=cfg.VAL_SIZE, seed=cfg.SEED)

    val_transform = build_preprocessed_val_transform(patch_size=cfg.PATCH_SIZE)
    _, val_loader = build_loaders(
        train_items=val_items,  # dummy, not used
        val_items=val_items,
        train_transform=val_transform,
        val_transform=val_transform,
        batch_size=8,
        num_workers=2,
        cache_rate_train=0.0,
        cache_rate_val=1.0,
        seed=cfg.SEED,
        pin_memory=cfg.PIN_MEMORY,
        collate_fn=pad_list_data_collate,
    )

    post_pred = Compose([EnsureType(), AsDiscrete(argmax=True)])

    images_shown = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if images_shown >= NUM_IMAGES:
                break
            images = batch["image"].to(device)
            labels = batch["label"]
            logits = model(images)
            preds = torch.stack([post_pred(x) for x in
                                 torch.unbind(logits, dim=0)])
            remaining = NUM_IMAGES - images_shown
            images = images[:remaining]
            labels = labels[:remaining]
            preds = preds[:remaining]
            visualize_batch(images, labels, preds, batch_idx, OUTPUT_DIR)
            images_shown += images.shape[0]

    print(f"\nDone! Saved {min(images_shown, NUM_IMAGES)} visualizations to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
