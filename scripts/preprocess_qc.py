from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
from monai.data.utils import affine_to_spacing
from monai.transforms import Compose, EnsureChannelFirstd, EnsureTyped, LoadImaged

# Allow the script to be executed from the repository root without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.load_data import build_acdc_list
from src.transforms import build_preprocess_transform


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the preprocessing quality-control run."""
    parser = argparse.ArgumentParser(
        description=(
            "Run quality-control checks on the deterministic preprocessing pipeline. "
            "The report is meant to answer: did preprocessing preserve geometry, labels, and "
            "reasonable tensor sizes before you spend time training?"
        )
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of ED/ES samples to inspect.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when sampling a subset of cases.",
    )
    parser.add_argument(
        "--target-spacing",
        type=float,
        nargs=3,
        default=(1.25, 1.25, -1.0),
        metavar=("SX", "SY", "SZ"),
        help="Spacing passed to build_preprocess_transform. Use -1 for native z spacing.",
    )
    parser.add_argument(
        "--pad-size",
        type=int,
        nargs=3,
        default=(192, 192, 16),
        metavar=("PX", "PY", "PZ"),
        help="Padding size passed to build_preprocess_transform.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "artifacts" / "preprocess_qc_report.json",
        help="Where to write the JSON report.",
    )
    return parser.parse_args()


def build_loader_transform() -> Compose:
    """Build the raw loading transform used as the pre-preprocessing reference."""
    # This transform loads the raw pair without resampling so we can compare
    # original geometry against the processed output from the real pipeline.
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            EnsureTyped(keys=["image", "label"]),
        ]
    )


def to_spatial_shape(tensor: Any) -> list[int]:
    """Extract the spatial dimensions only, dropping the leading channel axis."""
    return [int(dim) for dim in tensor.shape[1:]]


def to_spacing(tensor: Any) -> list[float]:
    """Recover voxel spacing from the tensor affine for report generation."""
    spacing = affine_to_spacing(tensor.affine, r=3)
    return [round(float(x), 4) for x in spacing.tolist()]


def to_unique_labels(tensor: Any) -> list[float]:
    """Return sorted label values while smoothing tiny floating-point noise."""
    values = np.unique(np.asarray(tensor))
    rounded_values: list[float] = []
    for value in values.tolist():
        # Labels should stay discrete after nearest-neighbor interpolation.
        # Rounding tiny floating-point noise keeps the report readable.
        rounded = round(float(value), 6)
        if abs(rounded - round(rounded)) < 1e-6:
            rounded = int(round(rounded))
        rounded_values.append(rounded)
    return rounded_values


def to_label_bbox_shape(tensor: Any) -> list[int]:
    """Measure the foreground label bounding-box size in channel-free coordinates."""
    label = np.asarray(tensor)
    if label.ndim == 4:
        label = label[0]

    foreground = np.argwhere(label > 0)
    if foreground.size == 0:
        return [0, 0, 0]

    bbox_min = foreground.min(axis=0)
    bbox_max = foreground.max(axis=0)
    return [int(x) for x in (bbox_max - bbox_min + 1)]


def label_integrity_ok(unique_values: list[float]) -> bool:
    """Check that label values remain effectively discrete after preprocessing."""
    return all(abs(float(v) - round(float(v))) < 1e-6 for v in unique_values)


def summarize_numeric_vectors(vectors: list[list[float]]) -> dict[str, list[float]]:
    """Compute min, median, and max summaries for shape or spacing vectors."""
    array = np.asarray(vectors, dtype=float)
    return {
        "min": np.round(array.min(axis=0), 4).tolist(),
        "median": np.round(np.median(array, axis=0), 4).tolist(),
        "max": np.round(array.max(axis=0), 4).tolist(),
    }


def inspect_case(item: dict[str, str], raw_transform: Compose, preprocess_transform: Compose) -> dict[str, Any]:
    """Compare one case before and after preprocessing and collect QC signals."""
    raw = raw_transform(item)
    processed = preprocess_transform(item)

    raw_image = raw["image"]
    raw_label = raw["label"]
    processed_image = processed["image"]
    processed_label = processed["label"]

    raw_label_values = to_unique_labels(raw_label)
    processed_label_values = to_unique_labels(processed_label)
    processed_label_bbox_shape = to_label_bbox_shape(processed_label)

    warnings: list[str] = []
    if to_spatial_shape(processed_image) != to_spatial_shape(processed_label):
        warnings.append("processed image/label shapes differ")
    if raw_label_values != processed_label_values:
        warnings.append("label set changed after preprocessing")
    if not label_integrity_ok(processed_label_values):
        warnings.append("processed labels are not discrete")

    processed_label_np = np.asarray(processed_label)
    foreground_fraction = float((processed_label_np > 0).mean())

    return {
        "patient": item["patient"],
        "phase": item["phase"],
        "image_path": item["image"],
        "label_path": item["label"],
        "raw_image_shape": to_spatial_shape(raw_image),
        "raw_label_shape": to_spatial_shape(raw_label),
        "raw_spacing": to_spacing(raw_image),
        "processed_image_shape": to_spatial_shape(processed_image),
        "processed_label_shape": to_spatial_shape(processed_label),
        "processed_spacing": to_spacing(processed_image),
        "raw_label_values": raw_label_values,
        "processed_label_values": processed_label_values,
        "processed_label_bbox_shape": processed_label_bbox_shape,
        # This is a quick proxy for crop aggressiveness and class imbalance.
        # Very small values usually mean the crop/pad window is too loose.
        "processed_foreground_fraction": round(foreground_fraction, 6),
        "warnings": warnings,
    }


def build_summary(case_reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-case QC results into dataset-level summary statistics."""
    raw_shapes = [case["raw_image_shape"] for case in case_reports]
    processed_shapes = [case["processed_image_shape"] for case in case_reports]
    raw_spacings = [case["raw_spacing"] for case in case_reports]
    processed_spacings = [case["processed_spacing"] for case in case_reports]
    processed_bbox_shapes = [case["processed_label_bbox_shape"] for case in case_reports]
    foreground_fractions = [case["processed_foreground_fraction"] for case in case_reports]

    warning_count = sum(1 for case in case_reports if case["warnings"])
    label_sets = sorted({tuple(case["processed_label_values"]) for case in case_reports})

    return {
        "num_cases": len(case_reports),
        "num_cases_with_warnings": warning_count,
        "raw_shape_summary": summarize_numeric_vectors(raw_shapes),
        "processed_shape_summary": summarize_numeric_vectors(processed_shapes),
        "raw_spacing_summary": summarize_numeric_vectors(raw_spacings),
        "processed_spacing_summary": summarize_numeric_vectors(processed_spacings),
        "processed_label_bbox_shape_summary": summarize_numeric_vectors(processed_bbox_shapes),
        "processed_foreground_fraction": {
            "min": round(float(np.min(foreground_fractions)), 6),
            "median": round(float(np.median(foreground_fractions)), 6),
            "max": round(float(np.max(foreground_fractions)), 6),
        },
        "processed_label_sets": [list(label_set) for label_set in label_sets],
    }


def print_summary(summary: dict[str, Any], case_reports: list[dict[str, Any]]) -> None:
    """Print the compact terminal summary and highlight warning cases."""
    print(f"Inspected cases: {summary['num_cases']}")
    print(f"Cases with warnings: {summary['num_cases_with_warnings']}")
    print(f"Raw spacing summary: {summary['raw_spacing_summary']}")
    print(f"Processed spacing summary: {summary['processed_spacing_summary']}")
    print(f"Raw shape summary: {summary['raw_shape_summary']}")
    print(f"Processed shape summary: {summary['processed_shape_summary']}")
    print(f"Processed label bbox summary: {summary['processed_label_bbox_shape_summary']}")
    print(f"Processed foreground fraction: {summary['processed_foreground_fraction']}")
    print(f"Observed processed label sets: {summary['processed_label_sets']}")

    warning_cases = [case for case in case_reports if case["warnings"]]
    if warning_cases:
        print("\nWarning cases:")
        for case in warning_cases:
            print(
                f"- {case['patient']} {case['phase']}: "
                f"{'; '.join(case['warnings'])}"
            )


def main() -> None:
    """Run the preprocessing QC workflow and save the JSON report."""
    args = parse_args()
    items = build_acdc_list()
    if not items:
        raise SystemExit("No ACDC items were found. Check config.py and dataset availability.")

    rng = random.Random(args.seed)
    sample_size = min(args.limit, len(items))
    sampled_items = rng.sample(items, k=sample_size)

    raw_transform = build_loader_transform()
    preprocess_transform = build_preprocess_transform(
        target_spacing=tuple(args.target_spacing),
        pad_size=tuple(args.pad_size),
    )

    case_reports: list[dict[str, Any]] = []
    for item in sampled_items:
        case_reports.append(inspect_case(item, raw_transform, preprocess_transform))

    summary = build_summary(case_reports)
    report = {
        "config": {
            "limit": sample_size,
            "seed": args.seed,
            "target_spacing": list(args.target_spacing),
            "pad_size": list(args.pad_size),
        },
        "summary": summary,
        "cases": case_reports,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))

    print_summary(summary, case_reports)
    print(f"\nSaved report to: {args.output}")


if __name__ == "__main__":
    main()
