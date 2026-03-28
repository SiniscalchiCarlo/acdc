from __future__ import annotations

import json
import sys
import time
from pathlib import Path
import os
import tempfile
import shutil

from monai.data import pad_list_data_collate
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from monai.networks.nets import UNet
from tqdm import tqdm
from monai.losses import DiceLoss

# Allow the script to be executed from the repository root without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import train_baseline_2d_config as cfg
import nils_training_config as cfg_n

from src.load_data_2D import (
    build_loaders,
    build_preprocessed_2d_list,
    load_preprocessed_2d_manifest,
    split_by_patient,
)
from src.transforms_2D import (
    build_preprocessed_train_transform,
    build_preprocessed_val_transform,
)

def main() -> None:
    """Data loader from basline model"""

    manifest = load_preprocessed_2d_manifest(cfg.PREPROCESSED_ROOT)
    items = build_preprocessed_2d_list(cfg.PREPROCESSED_ROOT)
    train_items, val_items = split_by_patient(items, n_splits=cfg.N_SPLITS, fold=cfg.FOLD)

    
    train_transform = build_preprocessed_train_transform(
        patch_size=cfg.PATCH_SIZE,
    )
    val_transform = build_preprocessed_val_transform(
        patch_size=cfg.PATCH_SIZE,
    )
    collate_fn = pad_list_data_collate

    train_loader, val_loader = build_loaders(
        train_items=train_items,
        val_items=val_items,
        train_transform=train_transform,
        val_transform=val_transform,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        cache_rate_train=cfg.CACHE_RATE_TRAIN,
        cache_rate_val=cfg.CACHE_RATE_VAL,
        seed=cfg.SEED,
        pin_memory=cfg.PIN_MEMORY,
        collate_fn=collate_fn,
    )
    print(f"Train loader: {len(train_loader.dataset)} samples in {len(train_loader)} batches")
    print(f"Validation loader: {len(val_loader.dataset)} samples in {len(val_loader)} batches")

    # Build model, optimizer, loss and simple training/validation loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(
        spatial_dims=2,
        in_channels=cfg_n.MODEL_IN_CHANNELS,
        out_channels=cfg_n.MODEL_OUT_CHANNELS,
        channels=tuple(cfg_n.MODEL_CHANNELS),
        strides=tuple(cfg_n.MODEL_STRIDES),
        num_res_units=int(cfg_n.MODEL_NUM_RES_UNITS),
    ).to(device)

    criterion = DiceLoss(
        include_background=cfg_n.DICE_INCLUDE_BACKGROUND,
        to_onehot_y=cfg_n.DICE_TO_ONEHOT_Y,
        softmax=cfg_n.DICE_SOFTMAX,
        reduction=cfg_n.DICE_REDUCTION,
    )
    optimizer = optim.AdamW(model.parameters(), lr=cfg_n.LR, weight_decay=cfg_n.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg_n.SCHEDULER_FACTOR,
        patience=cfg_n.SCHEDULER_PATIENCE,
        min_lr=cfg_n.SCHEDULER_MIN_LR,
    )

    def _prepare_labels(labels: torch.Tensor) -> torch.Tensor:
        # Ensure labels are integer class indices with shape [B, 1, H, W].
        if labels.ndim == 3:
            labels = labels.unsqueeze(1)
        elif labels.ndim == 4 and labels.shape[1] != 1:
            if labels.shape[1] > 1:
                labels = labels.argmax(dim=1, keepdim=True)
        return labels.long()

    def train_one_epoch(loader):
        model.train()
        total_loss = 0.0
        for batch in tqdm(loader, desc="Train", leave=False):
            images = batch["image"].to(device)
            labels = _prepare_labels(batch["label"].to(device))
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def validate(loader):
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(loader, desc="Val", leave=False):
                images = batch["image"].to(device)
                labels = _prepare_labels(batch["label"].to(device))
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
        return total_loss / len(loader)

    best_val = float("inf")
    for epoch in range(cfg_n.EPOCHS):
        t0 = time.time()
        train_loss = train_one_epoch(train_loader)
        val_loss = validate(val_loader) if (epoch + 1) % cfg_n.VAL_EVERY == 0 else None
        elapsed = time.time() - t0
        if val_loss is not None:
            print(f"Epoch {epoch+1}/{cfg_n.EPOCHS} - train_loss={train_loss:.4f} val_loss={val_loss:.4f} time={elapsed:.1f}s")
            # step scheduler on validation loss
            try:
                scheduler.step(val_loss)
            except Exception:
                pass
            if val_loss < best_val:
                best_val = val_loss
                out_path = cfg_n.MODEL_OUTPUT if getattr(cfg_n, "MODEL_OUTPUT", None) is not None else cfg.MODEL_OUTPUT
                out_path = Path(out_path)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                # Try atomic save in the target directory, fall back to current working directory if that fails
                try:
                    with tempfile.NamedTemporaryFile(dir=str(out_path.parent), delete=False) as tmpf:
                        tmp_path = Path(tmpf.name)
                    torch.save(model.state_dict(), str(tmp_path))
                    shutil.move(str(tmp_path), str(out_path))
                    globals()["last_saved_model"] = out_path
                except Exception as e:
                    print(f"Failed to save model to {out_path}: {e}")
                    fallback = Path.cwd() / "model_best.pth"
                    try:
                        torch.save(model.state_dict(), str(fallback))
                        print(f"Saved model to fallback path: {fallback}")
                        globals()["last_saved_model"] = fallback
                    except Exception as e2:
                        print(f"Fallback save also failed: {e2}")
                        raise
            # log current LR
            current_lr = optimizer.param_groups[0].get("lr", None)
            print(f"Current LR: {current_lr}")
        else:
            print(f"Epoch {epoch+1}/{cfg_n.EPOCHS} - train_loss={train_loss:.4f} time={elapsed:.1f}s")


if __name__ == "__main__":
    main()

    # After training, only report where the model was saved.
    final_model_path = globals().get(
        "last_saved_model",
        (cfg_n.MODEL_OUTPUT if getattr(cfg_n, "MODEL_OUTPUT", None) is not None else cfg.MODEL_OUTPUT),
    )
    print(f"Saved final model to: {final_model_path}")
