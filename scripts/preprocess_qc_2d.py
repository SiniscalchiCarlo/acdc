"""
Very simple QC script for the 2D preprocessing.

It checks a small set of slices, saves a few figures, and writes a tiny JSON
report with only the information you usually look at first.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from monai.transforms import Compose, EnsureChannelFirstd, EnsureTyped, LoadImaged

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import (
    foreground_margin_2d,
    patch_size_2d,
    preprocess_qc_limit_2d,
    preprocess_qc_output_figures_2d,
    preprocess_qc_output_json_2d,
    seed_2d,
    target_spacing_2d,
    use_foreground_crop_2d,
)
from src.load_data_2D import build_acdc_list
from src.transforms_2D import build_train_transform, build_val_transform

LIMIT = preprocess_qc_limit_2d
SEED = seed_2d
TARGET_SPACING = target_spacing_2d
PATCH_SIZE = patch_size_2d
USE_FOREGROUND_CROP = use_foreground_crop_2d
FOREGROUND_MARGIN = foreground_margin_2d

OUTPUT_JSON = Path(preprocess_qc_output_json_2d)
OUTPUT_FIGURES = Path(preprocess_qc_output_figures_2d)


def save_figure(
    output_path: Path,
    title: str,
    raw_image: np.ndarray,
    raw_label: np.ndarray,
    processed_image: np.ndarray,
    processed_label: np.ndarray,
    aug_image: np.ndarray,
    aug_label: np.ndarray,
) -> None:
    """Save one raw / processed / augmented comparison figure."""
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    figure.suptitle(title)

    panels = [
        (axes[0], raw_image, raw_label, "Raw"),
        (axes[1], processed_image, processed_label, "Processed"),
        (axes[2], aug_image, aug_label, "Augmented"),
    ]

    for ax, image, label, panel_title in panels:
        ax.imshow(image, cmap="gray", origin="lower")
        ax.imshow(np.ma.masked_where(label == 0, label), cmap="jet", alpha=0.35, origin="lower")
        ax.set_title(panel_title)
        ax.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Run a very small and readable preprocessing QC."""
    items = build_acdc_list()
    sampled_items = random.Random(SEED).sample(items, k=min(LIMIT, len(items)))

    raw_transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            EnsureTyped(keys=["image", "label"]),
        ]
    )

    preprocess_transform = build_val_transform(
        target_spacing=TARGET_SPACING,
        patch_size=PATCH_SIZE,
        use_foreground_crop=USE_FOREGROUND_CROP,
        foreground_margin=FOREGROUND_MARGIN,
    )
    train_transform = build_train_transform(
        target_spacing=TARGET_SPACING,
        patch_size=PATCH_SIZE,
        use_foreground_crop=USE_FOREGROUND_CROP,
        foreground_margin=FOREGROUND_MARGIN,
    )

    warnings = 0
    reports: list[dict[str, object]] = []
    raw_heights: list[int] = []
    raw_widths: list[int] = []
    processed_heights: list[int] = []
    processed_widths: list[int] = []
    processed_image_means: list[float] = []
    processed_image_stds: list[float] = []
    processed_foreground_fractions: list[float] = []

    for index, item in enumerate(sampled_items):
        raw = raw_transform(item)
        processed = preprocess_transform(item)
        augmented = train_transform(item)

        raw_image_volume = np.asarray(raw["image"])[0]
        raw_label_volume = np.asarray(raw["label"])[0]
        slice_idx = min(int(item["slice_idx"]), raw_image_volume.shape[-1] - 1)

        raw_image = raw_image_volume[..., slice_idx]
        raw_label = raw_label_volume[..., slice_idx]
        processed_image = np.asarray(processed["image"])[0]
        processed_label = np.asarray(processed["label"])[0]
        aug_image = np.asarray(augmented["image"])[0]
        aug_label = np.asarray(augmented["label"])[0]

        case_warnings: list[str] = []
        if processed_image.shape != processed_label.shape:
            case_warnings.append("image/label shape mismatch")
        if processed_image.shape[0] < PATCH_SIZE[0] or processed_image.shape[1] < PATCH_SIZE[1]:
            case_warnings.append("processed image is smaller than pad size")
        if processed_label.shape[0] < PATCH_SIZE[0] or processed_label.shape[1] < PATCH_SIZE[1]:
            case_warnings.append("processed label is smaller than pad size")
        if sorted(np.unique(processed_label).astype(int).tolist()) != sorted(np.unique(raw_label).astype(int).tolist()):
            case_warnings.append("label values changed")
        if np.count_nonzero(raw_label) > 0 and np.count_nonzero(processed_label) == 0:
            case_warnings.append("foreground disappeared")
        if any(label not in {0, 1, 2, 3} for label in np.unique(aug_label).astype(int).tolist()):
            case_warnings.append("augmented labels invalid")

        if case_warnings:
            warnings += 1

        raw_heights.append(int(raw_image.shape[0]))
        raw_widths.append(int(raw_image.shape[1]))
        processed_heights.append(int(processed_image.shape[0]))
        processed_widths.append(int(processed_image.shape[1]))
        processed_image_means.append(float(np.mean(processed_image)))
        processed_image_stds.append(float(np.std(processed_image)))
        processed_foreground_fractions.append(float((processed_label > 0).mean()))

        reports.append(
            {
                "patient": item["patient"],
                "phase": item["phase"],
                "slice_idx": int(item["slice_idx"]),
                "warnings": case_warnings,
            }
        )

        if index < 6:
            save_figure(
                output_path=OUTPUT_FIGURES / f"{item['patient']}_{item['phase']}_slice_{int(item['slice_idx']):02d}.png",
                title=f"{item['patient']} {item['phase']} slice={item['slice_idx']}",
                raw_image=raw_image,
                raw_label=raw_label,
                processed_image=processed_image,
                processed_label=processed_label,
                aug_image=aug_image,
                aug_label=aug_label,
            )

    summary = {
        "sampled_slices": len(sampled_items),
        "slices_with_warnings": warnings,
        "raw_height_min": min(raw_heights),
        "raw_height_avg": round(float(np.mean(raw_heights)), 3),
        "raw_height_max": max(raw_heights),
        "raw_width_min": min(raw_widths),
        "raw_width_avg": round(float(np.mean(raw_widths)), 3),
        "raw_width_max": max(raw_widths),
        "processed_height_min": min(processed_heights),
        "processed_height_avg": round(float(np.mean(processed_heights)), 3),
        "processed_height_max": max(processed_heights),
        "processed_width_min": min(processed_widths),
        "processed_width_avg": round(float(np.mean(processed_widths)), 3),
        "processed_width_max": max(processed_widths),
        "processed_image_mean_min": round(min(processed_image_means), 6),
        "processed_image_mean_avg": round(float(np.mean(processed_image_means)), 6),
        "processed_image_mean_max": round(max(processed_image_means), 6),
        "processed_image_std_min": round(min(processed_image_stds), 6),
        "processed_image_std_avg": round(float(np.mean(processed_image_stds)), 6),
        "processed_image_std_max": round(max(processed_image_stds), 6),
        "processed_foreground_fraction_min": round(min(processed_foreground_fractions), 6),
        "processed_foreground_fraction_avg": round(float(np.mean(processed_foreground_fractions)), 6),
        "processed_foreground_fraction_max": round(max(processed_foreground_fractions), 6),
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps({"summary": summary, "reports": reports}, indent=2))

    print(f"Sampled slices: {summary['sampled_slices']}")
    print(f"Slices with warnings: {summary['slices_with_warnings']}")
    print(
        "Raw height min/avg/max:",
        summary["raw_height_min"],
        summary["raw_height_avg"],
        summary["raw_height_max"],
    )
    print(
        "Raw width min/avg/max:",
        summary["raw_width_min"],
        summary["raw_width_avg"],
        summary["raw_width_max"],
    )
    print(
        "Processed height min/avg/max:",
        summary["processed_height_min"],
        summary["processed_height_avg"],
        summary["processed_height_max"],
    )
    print(
        "Processed width min/avg/max:",
        summary["processed_width_min"],
        summary["processed_width_avg"],
        summary["processed_width_max"],
    )
    print(
        "Processed image mean min/avg/max:",
        summary["processed_image_mean_min"],
        summary["processed_image_mean_avg"],
        summary["processed_image_mean_max"],
    )
    print(
        "Processed image std min/avg/max:",
        summary["processed_image_std_min"],
        summary["processed_image_std_avg"],
        summary["processed_image_std_max"],
    )
    print(
        "Processed foreground fraction min/avg/max:",
        summary["processed_foreground_fraction_min"],
        summary["processed_foreground_fraction_avg"],
        summary["processed_foreground_fraction_max"],
    )
    print(f"Saved report to: {OUTPUT_JSON}")
    print(f"Saved figures to: {OUTPUT_FIGURES}")


if __name__ == "__main__":
    main()
