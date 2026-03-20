from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from monai.data import pad_list_data_collate
import numpy as np

# Allow the script to be executed from the repository root without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import train_baseline_2d_config as cfg
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
    """Train and validate a reproducible 2D baseline using config-only settings."""

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



if __name__ == "__main__":
    main()
