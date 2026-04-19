from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from monai.data import CacheDataset, DataLoader
from monai.utils import set_determinism
from sklearn.model_selection import GroupShuffleSplit

from config import dataset_path


def parse_info_cfg(info_path: Path):
    """
    Parse Info.cfg ACDC:
    extracts ED e ES (frame indices).
    """
    txt = info_path.read_text(errors="ignore")
    m_ed = re.search(r"\bED\s*:\s*(\d+)", txt)
    m_es = re.search(r"\bES\s*:\s*(\d+)", txt)
    ed = int(m_ed.group(1)) if m_ed else None
    es = int(m_es.group(1)) if m_es else None
    return ed, es


def split_by_patient(
    items: list[dict[str, Any]],
    val_size: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split slice-level items by patient so slices never leak across train and validation."""

    groups = np.asarray([item["patient"] for item in items])

    indices = np.arange(len(items))
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_idx, val_idx = next(splitter.split(indices, y=None, groups=groups))

    train_items = [items[idx] for idx in train_idx]
    val_items = [items[idx] for idx in val_idx]
    return train_items, val_items


def build_acdc_list(
    dataset_root: str | Path | None = None,
    include_background_slices: bool = True,
    min_label_pixels: int = 1,
) -> list[dict[str, Any]]:
    """
    Build a slice-level ACDC list for 2D training.

    Each item corresponds to one 2D slice from an ED or ES volume. Splitting can
    still happen by patient because the patient id is attached to every slice.
    """

    root = Path(dataset_path if dataset_root is None else dataset_root)
    patients = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("patient")])

    items: list[dict[str, Any]] = []

    for patient_dir in patients:
        ed_frame, es_frame = parse_info_cfg(patient_dir / "Info.cfg")
        for frame_idx, phase in ((ed_frame, "ED"), (es_frame, "ES")):
            items.extend(
                _build_slice_items(
                    patient_dir=patient_dir,
                    frame_idx=frame_idx,
                    phase=phase,
                    include_background_slices=include_background_slices,
                    min_label_pixels=min_label_pixels,
                )
            )

    if not items:
        raise RuntimeError("No 2D slice items were built from the configured dataset path.")

    return items


def build_preprocessed_dataset_list(
    preprocessed_root: str | Path,
) -> list[dict[str, Any]]:
    """Load a preprocessed slice dataset manifest from disk."""
    manifest = load_preprocessed_manifest(preprocessed_root=preprocessed_root)
    items = manifest.get("items", [])
    if not items:
        raise RuntimeError(f"No items found in preprocessed manifest: {Path(preprocessed_root) / 'manifest.json'}")

    root = Path(preprocessed_root)
    normalized_items: list[dict[str, Any]] = []
    for item in items:
        item_copy = dict(item)
        sample_path = Path(item_copy["sample"])
        if not sample_path.is_absolute():
            sample_path = root / sample_path
        item_copy["sample"] = str(sample_path)
        normalized_items.append(item_copy)
    return normalized_items


def load_preprocessed_manifest(
    preprocessed_root: str | Path,
) -> dict[str, Any]:
    """Load and return the full preprocessed manifest, including stored config."""
    root = Path(preprocessed_root)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Preprocessed manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text())
    if "items" not in manifest:
        raise RuntimeError(f"Malformed preprocessed manifest: missing 'items' in {manifest_path}")
    return manifest


def _build_slice_items(
    patient_dir: Path,
    frame_idx: int,
    phase: str,
    include_background_slices: bool,
    min_label_pixels: int,
) -> list[dict[str, Any]]:
    image_path = patient_dir / f"{patient_dir.name}_frame{frame_idx:02d}.nii.gz"
    label_path = patient_dir / f"{patient_dir.name}_frame{frame_idx:02d}_gt.nii.gz"

    label_nii = nib.load(str(label_path))
    label_data = np.asarray(label_nii.dataobj)

    slice_items: list[dict[str, Any]] = []
    for slice_idx in range(label_data.shape[-1]):
        label_slice = label_data[..., slice_idx]
        label_pixels = int(np.count_nonzero(label_slice))
        has_foreground = label_pixels >= min_label_pixels

        if not include_background_slices and not has_foreground:
            continue

        slice_items.append(
            {
                "image": str(image_path),
                "label": str(label_path),
                "patient": patient_dir.name,
                "phase": phase,
                "frame": int(frame_idx),
                "slice_idx": int(slice_idx),
                "has_foreground": has_foreground,
                "label_pixels": label_pixels,
            }
        )

    return slice_items


def build_loaders(
    train_items: list[dict[str, Any]],
    val_items: list[dict[str, Any]],
    train_transform,
    val_transform,
    batch_size: int = 8,
    num_workers: int = 4,
    cache_rate_train: float = 1.0,
    cache_rate_val: float = 1.0,
    seed: int = 42,
    pin_memory: bool = True,
    collate_fn=None,
) -> tuple[DataLoader, DataLoader]:
    """Create MONAI cache datasets and data loaders for 2D slice training."""
    set_determinism(seed=seed)

    train_ds = CacheDataset(
        data=train_items,
        transform=train_transform,
        cache_rate=cache_rate_train,
        num_workers=num_workers,
    )
    val_ds = CacheDataset(
        data=val_items,
        transform=val_transform,
        cache_rate=cache_rate_val,
        num_workers=num_workers,
    )

    persistent_workers = num_workers > 0
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader
