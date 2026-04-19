from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Allow the script to be executed from the repository root without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import (
    include_background_slices_2d,
    min_label_pixels_2d,
    patch_size_2d,
    preprocessing_mode,
    preprocessed_data_path,
    target_spacing_2d,
)
from src.load_data_2D import build_acdc_list
from src.pipeline import preprocessing_mode_uses_triplet_slices
from src.transforms_2D import build_preprocessing_transform

TARGET_SPACING = target_spacing_2d
PATCH_SIZE = patch_size_2d
INCLUDE_BACKGROUND_SLICES = include_background_slices_2d
MIN_LABEL_PIXELS = min_label_pixels_2d
PREPROCESSING_MODE = preprocessing_mode


def main() -> None:
    """Precompute deterministic preprocessing for the full slice dataset."""
    items = build_acdc_list(
        include_background_slices=INCLUDE_BACKGROUND_SLICES,
        min_label_pixels=MIN_LABEL_PIXELS,
    )
    transform = build_preprocessing_transform(
        target_spacing=TARGET_SPACING,
        patch_size=PATCH_SIZE,
        preprocessing_mode=PREPROCESSING_MODE,
    )

    output_dir = Path(preprocessed_data_path)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    manifest_items: list[dict[str, object]] = []
    for item in tqdm(items, desc="Preprocessing", unit="slice"):
        transformed = transform(item)
        # FIX: Do not index [0] for images, as 2.5D requires all 3 channels
        image = np.asarray(transformed["image"]).astype(np.float32, copy=False)
        label = np.asarray(transformed["label"]).astype(np.int16, copy=False)

        sample_name = f"{item['patient']}_{item['phase']}_frame{item['frame']:02d}_slice{item['slice_idx']:02d}.npz"
        sample_path = samples_dir / sample_name
        np.savez_compressed(sample_path, image=image, label=label)

        manifest_items.append(
            {
                "sample": str(sample_path.relative_to(output_dir)),
                "patient": item["patient"],
                "phase": item["phase"],
                "frame": int(item["frame"]),
                "slice_idx": int(item["slice_idx"]),
                "has_foreground": bool(item["has_foreground"]),
                "label_pixels": int(item["label_pixels"]),
            }
        )

    manifest = {
        "config": {
            "target_spacing": list(TARGET_SPACING),
            "patch_size": list(PATCH_SIZE),
            "include_background_slices": INCLUDE_BACKGROUND_SLICES,
            "min_label_pixels": MIN_LABEL_PIXELS,
            "output_dir": str(output_dir),
            "preprocessing_mode": PREPROCESSING_MODE,
            "triplet_slices": preprocessing_mode_uses_triplet_slices(PREPROCESSING_MODE),
        },
        "num_items": len(manifest_items),
        "items": manifest_items,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"Saved {len(manifest_items)} preprocessed slices to: {samples_dir}")
    print(f"Saved manifest to: {manifest_path}")


if __name__ == "__main__":
    main()
