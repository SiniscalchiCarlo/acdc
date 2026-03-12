from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from monai.data.utils import affine_to_spacing
from monai.transforms import Compose, EnsureChannelFirstd, EnsureTyped, LoadImaged

# Allow the script to be executed from the repository root without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.load_data_2D import build_acdc_list
from src.transforms_2D import build_train_transform, build_val_transform


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for 2D preprocessing quality control."""
    parser = argparse.ArgumentParser(
        description=(
            "Run quantitative and visual QC checks on the 2D preprocessing pipeline. "
            "The report covers geometry, labels, slice distribution, and augmentation integrity."
        )
    )
    parser.add_argument("--limit", type=int, default=100, help="Maximum number of slices to inspect.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used to sample slices.")
    parser.add_argument(
        "--target-spacing",
        type=float,
        nargs=3,
        default=(1.25, 1.25, -1.0),
        metavar=("SX", "SY", "SZ"),
        help="Spacing passed to the 2D preprocessing transforms.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(192, 192),
        metavar=("PX", "PY"),
        help="Final 2D size passed to the preprocessing transforms.",
    )
    parser.add_argument(
        "--foreground-margin",
        type=int,
        default=16,
        help="Margin used when foreground-cropping the extracted 2D slice.",
    )
    parser.add_argument(
        "--include-background-slices",
        action="store_true",
        help="Include background-only slices in the QC sample pool.",
    )
    parser.add_argument(
        "--min-label-pixels",
        type=int,
        default=1,
        help="Minimum foreground pixels for a slice to be flagged as foreground.",
    )
    parser.add_argument(
        "--visual-limit",
        type=int,
        default=8,
        help="Maximum number of raw/processed/augmented figures to save.",
    )
    parser.add_argument(
        "--augmented-samples",
        type=int,
        default=2,
        help="Number of augmentation draws per slice for augmentation integrity checks.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "artifacts" / "preprocess_qc_report_2d.json",
        help="Where to write the JSON report.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "preprocess_visual_qc_2d",
        help="Directory where visual QC PNGs will be saved.",
    )
    return parser.parse_args()


def build_loader_transform() -> Compose:
    """Build the raw loading transform used as the pre-preprocessing reference."""
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            EnsureTyped(keys=["image", "label"]),
        ]
    )


def clamp_slice_idx(slice_idx: int, depth: int) -> int:
    """Clamp a requested z index to the valid range for the raw volume."""
    if depth < 1:
        raise ValueError("Depth must be >= 1")
    return max(0, min(int(slice_idx), depth - 1))


def to_unique_labels(array: np.ndarray) -> list[float]:
    """Return sorted label values while smoothing tiny floating-point noise."""
    values = np.unique(array)
    rounded_values: list[float] = []
    for value in values.tolist():
        rounded = round(float(value), 6)
        if abs(rounded - round(rounded)) < 1e-6:
            rounded = int(round(rounded))
        rounded_values.append(rounded)
    return rounded_values


def label_integrity_ok(unique_values: list[float]) -> bool:
    """Check that label values remain effectively discrete after preprocessing."""
    return all(abs(float(v) - round(float(v))) < 1e-6 for v in unique_values)


def to_spacing(tensor: Any) -> list[float]:
    """Recover voxel spacing from the tensor affine for report generation."""
    spacing = affine_to_spacing(tensor.affine, r=3)
    return [round(float(x), 4) for x in spacing.tolist()]


def summarize_vectors(vectors: list[list[float]]) -> dict[str, list[float]]:
    """Compute min, median, and max summaries for vector-valued signals."""
    array = np.asarray(vectors, dtype=float)
    return {
        "min": np.round(array.min(axis=0), 4).tolist(),
        "median": np.round(np.median(array, axis=0), 4).tolist(),
        "max": np.round(array.max(axis=0), 4).tolist(),
    }


def summarize_scalars(values: list[float]) -> dict[str, float]:
    """Compute min, median, and max summaries for scalar signals."""
    array = np.asarray(values, dtype=float)
    return {
        "min": round(float(array.min()), 6),
        "median": round(float(np.median(array)), 6),
        "max": round(float(array.max()), 6),
    }


def normalize_for_display(image: np.ndarray) -> np.ndarray:
    """Map image intensities to [0, 1] for visualization."""
    lower = float(np.percentile(image, 1))
    upper = float(np.percentile(image, 99))
    if upper <= lower:
        return np.zeros_like(image, dtype=float)
    clipped = np.clip(image, lower, upper)
    return (clipped - lower) / (upper - lower)


def render_for_display(array_2d: np.ndarray) -> np.ndarray:
    """Apply a consistent orientation for display panels."""
    return np.flipud(array_2d.T)


def save_case_figure(
    item: dict[str, Any],
    raw_image: np.ndarray,
    raw_label: np.ndarray,
    processed_image: np.ndarray,
    processed_label: np.ndarray,
    augmented_image: np.ndarray,
    augmented_label: np.ndarray,
    output_path: Path,
) -> None:
    """Save a raw/processed/augmented visual QC panel for one 2D slice."""
    figure, axes = plt.subplots(2, 3, figsize=(14, 8))
    figure.suptitle(
        f"{item['patient']} {item['phase']} frame={item['frame']:02d} slice={item['slice_idx']}"
    )

    panels = [
        (axes[0, 0], raw_image, raw_label, "Raw overlay"),
        (axes[0, 1], processed_image, processed_label, "Processed overlay"),
        (axes[0, 2], augmented_image, augmented_label, "Augmented overlay"),
        (axes[1, 0], raw_image, None, "Raw image"),
        (axes[1, 1], processed_image, None, "Processed image"),
        (axes[1, 2], augmented_image, None, "Augmented image"),
    ]

    for ax, image, label, title in panels:
        ax.imshow(render_for_display(normalize_for_display(image)), cmap="gray", origin="lower")
        if label is not None:
            displayed_label = render_for_display(label)
            ax.imshow(np.ma.masked_where(displayed_label == 0, displayed_label), cmap="jet", alpha=0.35, origin="lower")
        ax.set_title(title)
        ax.axis("off")

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def inspect_case(
    item: dict[str, Any],
    raw_transform: Compose,
    preprocess_transform: Compose,
    train_transform: Compose,
    patch_size: tuple[int, int],
    augmented_samples: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compare one raw slice against deterministic preprocessing and augmented outputs."""
    raw = raw_transform(item)
    raw_image_volume = np.asarray(raw["image"])[0]
    raw_label_volume = np.asarray(raw["label"])[0]
    raw_slice_idx = clamp_slice_idx(item["slice_idx"], raw_image_volume.shape[-1])
    raw_image = raw_image_volume[..., raw_slice_idx]
    raw_label = raw_label_volume[..., raw_slice_idx]

    processed = preprocess_transform(item)
    processed_image = np.asarray(processed["image"])[0]
    processed_label = np.asarray(processed["label"])[0]

    raw_label_values = to_unique_labels(raw_label)
    processed_label_values = to_unique_labels(processed_label)

    warnings: list[str] = []
    if processed_image.shape != processed_label.shape:
        warnings.append("processed image/label shapes differ")
    if tuple(int(x) for x in processed_image.shape) != tuple(int(x) for x in patch_size):
        warnings.append("processed shape does not match patch size")
    if raw_label_values != processed_label_values:
        warnings.append("label set changed after preprocessing")
    if not label_integrity_ok(processed_label_values):
        warnings.append("processed labels are not discrete")
    if np.count_nonzero(raw_label) > 0 and np.count_nonzero(processed_label) == 0:
        warnings.append("foreground disappeared after preprocessing")
    if float(np.std(processed_image)) < 1e-4:
        warnings.append("processed image has near-zero intensity variance")

    augmentation_reports: list[dict[str, Any]] = []
    for aug_idx in range(1, augmented_samples + 1):
        augmented = train_transform(item)
        augmented_image = np.asarray(augmented["image"])[0]
        augmented_label = np.asarray(augmented["label"])[0]
        augmented_label_values = to_unique_labels(augmented_label)
        aug_warnings: list[str] = []

        if augmented_image.shape != augmented_label.shape:
            aug_warnings.append("augmented image/label shapes differ")
        if tuple(int(x) for x in augmented_image.shape) != tuple(int(x) for x in patch_size):
            aug_warnings.append("augmented shape does not match patch size")
        if not label_integrity_ok(augmented_label_values):
            aug_warnings.append("augmented labels are not discrete")
        if set(augmented_label_values) - {0, 1, 2, 3}:
            aug_warnings.append("augmented labels contain unexpected classes")
        if np.count_nonzero(raw_label) > 0 and np.count_nonzero(augmented_label) == 0:
            aug_warnings.append("augmentation removed all foreground")

        augmentation_reports.append(
            {
                "sample_idx": aug_idx,
                "image_shape": [int(x) for x in augmented_image.shape],
                "label_shape": [int(x) for x in augmented_label.shape],
                "label_values": augmented_label_values,
                "foreground_fraction": round(float((augmented_label > 0).mean()), 6),
                "warnings": aug_warnings,
            }
        )

    case_report = {
        "patient": item["patient"],
        "phase": item["phase"],
        "frame": int(item["frame"]),
        "slice_idx": int(item["slice_idx"]),
        "has_foreground": bool(item["has_foreground"]),
        "label_pixels": int(item["label_pixels"]),
        "raw_spacing": to_spacing(raw["image"]),
        "processed_image_shape": [int(x) for x in processed_image.shape],
        "processed_label_shape": [int(x) for x in processed_label.shape],
        "raw_label_values": raw_label_values,
        "processed_label_values": processed_label_values,
        "raw_foreground_fraction": round(float((raw_label > 0).mean()), 6),
        "processed_foreground_fraction": round(float((processed_label > 0).mean()), 6),
        "processed_mean": round(float(np.mean(processed_image)), 6),
        "processed_std": round(float(np.std(processed_image)), 6),
        "processed_p01": round(float(np.percentile(processed_image, 1)), 6),
        "processed_p99": round(float(np.percentile(processed_image, 99)), 6),
        "warnings": warnings,
    }
    return case_report, augmentation_reports


def build_summary(
    items: list[dict[str, Any]],
    case_reports: list[dict[str, Any]],
    augmentation_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate slice-level and augmentation-level QC results into summary statistics."""
    processed_shapes = [report["processed_image_shape"] for report in case_reports]
    raw_spacings = [report["raw_spacing"] for report in case_reports]
    foreground_fractions = [report["processed_foreground_fraction"] for report in case_reports]
    processed_std = [report["processed_std"] for report in case_reports]
    label_pixel_counts = [int(item["label_pixels"]) for item in items]

    num_foreground = sum(1 for item in items if item["has_foreground"])
    num_background = len(items) - num_foreground
    case_warning_count = sum(1 for report in case_reports if report["warnings"])
    aug_warning_count = sum(1 for report in augmentation_reports if report["warnings"])

    return {
        "num_inspected_slices": len(case_reports),
        "num_candidate_slices": len(items),
        "num_foreground_slices": num_foreground,
        "num_background_slices": num_background,
        "foreground_slice_fraction": round(num_foreground / max(len(items), 1), 6),
        "label_pixel_count_summary": summarize_scalars([float(x) for x in label_pixel_counts]),
        "raw_spacing_summary": summarize_vectors(raw_spacings),
        "processed_shape_summary": summarize_vectors(processed_shapes),
        "processed_foreground_fraction_summary": summarize_scalars(foreground_fractions),
        "processed_std_summary": summarize_scalars(processed_std),
        "observed_processed_label_sets": sorted(
            [list(label_set) for label_set in {tuple(report["processed_label_values"]) for report in case_reports}]
        ),
        "num_case_reports_with_warnings": case_warning_count,
        "num_augmentation_reports_with_warnings": aug_warning_count,
    }


def print_summary(summary: dict[str, Any], case_reports: list[dict[str, Any]], augmentation_reports: list[dict[str, Any]]) -> None:
    """Print a compact terminal summary and the first warning cases."""
    print(f"Candidate slices: {summary['num_candidate_slices']}")
    print(f"Foreground slices: {summary['num_foreground_slices']}")
    print(f"Background slices: {summary['num_background_slices']}")
    print(f"Inspected slices: {summary['num_inspected_slices']}")
    print(f"Raw spacing summary: {summary['raw_spacing_summary']}")
    print(f"Processed shape summary: {summary['processed_shape_summary']}")
    print(f"Processed foreground fraction summary: {summary['processed_foreground_fraction_summary']}")
    print(f"Processed std summary: {summary['processed_std_summary']}")
    print(f"Observed processed label sets: {summary['observed_processed_label_sets']}")
    print(f"Case warnings: {summary['num_case_reports_with_warnings']}")
    print(f"Augmentation warnings: {summary['num_augmentation_reports_with_warnings']}")

    first_case_warnings = [report for report in case_reports if report["warnings"]][:5]
    if first_case_warnings:
        print("\nCase warnings:")
        for report in first_case_warnings:
            print(
                f"- {report['patient']} {report['phase']} slice={report['slice_idx']}: "
                f"{'; '.join(report['warnings'])}"
            )

    first_aug_warnings = [report for report in augmentation_reports if report["warnings"]][:5]
    if first_aug_warnings:
        print("\nAugmentation warnings:")
        for report in first_aug_warnings:
            print(
                f"- sample={report['sample_idx']} warnings={'; '.join(report['warnings'])}"
            )


def main() -> None:
    """Run the 2D preprocessing QC workflow and save the JSON report."""
    args = parse_args()
    items = build_acdc_list(
        include_background_slices=args.include_background_slices,
        min_label_pixels=args.min_label_pixels,
    )
    if not items:
        raise SystemExit("No 2D ACDC items were found. Check config.py and dataset availability.")

    rng = random.Random(args.seed)
    sampled_items = rng.sample(items, k=min(args.limit, len(items)))

    raw_transform = build_loader_transform()
    preprocess_transform = build_val_transform(
        target_spacing=tuple(args.target_spacing),
        patch_size=tuple(args.patch_size),
        foreground_margin=args.foreground_margin,
    )
    train_transform = build_train_transform(
        target_spacing=tuple(args.target_spacing),
        patch_size=tuple(args.patch_size),
        foreground_margin=args.foreground_margin,
    )

    case_reports: list[dict[str, Any]] = []
    all_augmentation_reports: list[dict[str, Any]] = []

    for item in sampled_items:
        case_report, augmentation_reports = inspect_case(
            item=item,
            raw_transform=raw_transform,
            preprocess_transform=preprocess_transform,
            train_transform=train_transform,
            patch_size=tuple(args.patch_size),
            augmented_samples=args.augmented_samples,
        )
        case_reports.append(case_report)

        for augmentation_report in augmentation_reports:
            augmented_entry = dict(augmentation_report)
            augmented_entry["patient"] = item["patient"]
            augmented_entry["phase"] = item["phase"]
            augmented_entry["slice_idx"] = int(item["slice_idx"])
            all_augmentation_reports.append(augmented_entry)

    visual_items = sampled_items[: min(args.visual_limit, len(sampled_items))]
    for item in visual_items:
        raw = raw_transform(item)
        raw_image_volume = np.asarray(raw["image"])[0]
        raw_label_volume = np.asarray(raw["label"])[0]
        raw_slice_idx = clamp_slice_idx(item["slice_idx"], raw_image_volume.shape[-1])
        raw_image = raw_image_volume[..., raw_slice_idx]
        raw_label = raw_label_volume[..., raw_slice_idx]

        processed = preprocess_transform(item)
        processed_image = np.asarray(processed["image"])[0]
        processed_label = np.asarray(processed["label"])[0]

        augmented = train_transform(item)
        augmented_image = np.asarray(augmented["image"])[0]
        augmented_label = np.asarray(augmented["label"])[0]

        figure_path = args.figure_dir / f"{item['patient']}_{item['phase']}_slice_{item['slice_idx']:02d}.png"
        save_case_figure(
            item=item,
            raw_image=raw_image,
            raw_label=raw_label,
            processed_image=processed_image,
            processed_label=processed_label,
            augmented_image=augmented_image,
            augmented_label=augmented_label,
            output_path=figure_path,
        )

    summary = build_summary(items=items, case_reports=case_reports, augmentation_reports=all_augmentation_reports)
    report = {
        "config": {
            "limit": len(sampled_items),
            "seed": args.seed,
            "target_spacing": list(args.target_spacing),
            "patch_size": list(args.patch_size),
            "foreground_margin": args.foreground_margin,
            "include_background_slices": args.include_background_slices,
            "min_label_pixels": args.min_label_pixels,
            "augmented_samples": args.augmented_samples,
            "visual_limit": min(args.visual_limit, len(sampled_items)),
        },
        "summary": summary,
        "case_reports": case_reports,
        "augmentation_reports": all_augmentation_reports,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print_summary(summary, case_reports, all_augmentation_reports)
    print(f"\nSaved report to: {args.output}")
    print(f"Saved figures to: {args.figure_dir}")


if __name__ == "__main__":
    main()
