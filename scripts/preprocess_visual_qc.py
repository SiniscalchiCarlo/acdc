from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from monai.transforms import Compose, EnsureChannelFirstd, EnsureTyped, LoadImaged

# Allow the script to be executed from the repository root without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.load_data import build_acdc_list
from src.transforms import build_preprocess_transform, build_train_transform


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for preprocessing visualization quality control."""
    parser = argparse.ArgumentParser(
        description=(
            "Save raw-vs-preprocessed slice overlays so preprocessing can be checked visually "
            "before training."
        )
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=4,
        help="Maximum number of ED/ES cases to visualize.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "preprocess_visual_qc",
        help="Directory where PNG figures will be saved.",
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
        "--include-augmented",
        action="store_true",
        help="Also save one or more augmented samples from the training transform.",
    )
    parser.add_argument(
        "--num-augmented-samples",
        type=int,
        default=2,
        help="Number of random augmented samples to save per case when augmentation is enabled.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used to select which cases are visualized.",
    )
    return parser.parse_args()


def build_loader_transform() -> Compose:
    """Build the raw loading transform used for the pre-preprocessing view."""
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            EnsureTyped(keys=["image", "label"]),
        ]
    )


def to_numpy_volume(tensor) -> np.ndarray:
    """Convert a channel-first MONAI tensor or array to a plain 3D NumPy volume."""
    array = np.asarray(tensor)
    if array.ndim != 4:
        raise ValueError(f"Expected a 4D channel-first tensor, got shape {array.shape}.")
    return array[0]


def choose_slice_index(label_volume: np.ndarray) -> int:
    """Choose the slice with the largest foreground area to visualize cardiac anatomy."""
    foreground_per_slice = (label_volume > 0).sum(axis=(0, 1))
    if foreground_per_slice.max() == 0:
        return int(label_volume.shape[-1] // 2)
    return int(foreground_per_slice.argmax())


def normalize_for_display(image: np.ndarray) -> np.ndarray:
    """Map intensities to [0, 1] using robust percentiles for readable figures."""
    lower = float(np.percentile(image, 1))
    upper = float(np.percentile(image, 99))
    if upper <= lower:
        return np.zeros_like(image, dtype=float)
    clipped = np.clip(image, lower, upper)
    return (clipped - lower) / (upper - lower)


def format_case_metadata(image_volume: np.ndarray, item: dict[str, str]) -> str:
    """Build a compact label used in figure titles and filenames."""
    return f"{item['patient']} {item['phase']} shape={tuple(image_volume.shape)}"


def render_slice_for_display(slice_2d: np.ndarray) -> np.ndarray:
    """Convert array layout into a consistent plotting view for every panel."""
    return np.flipud(slice_2d.T)


def plot_overlay(ax, image_slice: np.ndarray, label_slice: np.ndarray, title: str, metadata: str) -> None:
    """Render grayscale anatomy with a transparent label overlay."""
    displayed_image = render_slice_for_display(normalize_for_display(image_slice))
    displayed_label = render_slice_for_display(label_slice)
    ax.imshow(displayed_image, cmap="gray", origin="lower")
    ax.imshow(np.ma.masked_where(displayed_label == 0, displayed_label), cmap="jet", alpha=0.35, origin="lower")
    ax.set_title(title)
    ax.text(0.02, 0.02, metadata, color="white", fontsize=8, transform=ax.transAxes, bbox={"facecolor": "black", "alpha": 0.5, "pad": 2})
    ax.axis("off")


def plot_image_only(ax, image_slice: np.ndarray, title: str, metadata: str) -> None:
    """Render a single image slice without a label overlay."""
    ax.imshow(render_slice_for_display(normalize_for_display(image_slice)), cmap="gray", origin="lower")
    ax.set_title(title)
    ax.text(0.02, 0.02, metadata, color="white", fontsize=8, transform=ax.transAxes, bbox={"facecolor": "black", "alpha": 0.5, "pad": 2})
    ax.axis("off")


def save_preprocess_figure(
    item: dict[str, str],
    raw: dict[str, object],
    processed: dict[str, object],
    output_dir: Path,
) -> Path:
    """Create and save a raw-vs-preprocessed overlay figure for one case."""
    raw_image = to_numpy_volume(raw["image"])
    raw_label = to_numpy_volume(raw["label"])
    processed_image = to_numpy_volume(processed["image"])
    processed_label = to_numpy_volume(processed["label"])

    raw_slice = choose_slice_index(raw_label)
    processed_slice = choose_slice_index(processed_label)

    figure, axes = plt.subplots(2, 2, figsize=(12, 10))
    figure.suptitle(format_case_metadata(processed_image, item))

    raw_metadata = f"slice={raw_slice}"
    processed_metadata = f"slice={processed_slice}"

    plot_overlay(
        axes[0, 0],
        raw_image[:, :, raw_slice],
        raw_label[:, :, raw_slice],
        "Raw image + label",
        raw_metadata,
    )
    plot_overlay(
        axes[0, 1],
        processed_image[:, :, processed_slice],
        processed_label[:, :, processed_slice],
        "Preprocessed image + label",
        processed_metadata,
    )

    # These image-only panels make it easier to notice contrast changes and crop boundaries
    # without the label overlay hiding subtle artifacts.
    plot_image_only(
        axes[1, 0],
        raw_image[:, :, raw_slice],
        "Raw image only",
        raw_metadata,
    )
    plot_image_only(
        axes[1, 1],
        processed_image[:, :, processed_slice],
        "Preprocessed image only",
        processed_metadata,
    )

    figure.tight_layout()

    output_path = output_dir / f"{item['patient']}_{item['phase']}_preprocess.png"
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output_path


def save_augmented_figure(
    item: dict[str, str],
    augmented: dict[str, object],
    sample_idx: int,
    output_dir: Path,
) -> Path:
    """Create and save an overlay figure for one random augmented sample."""
    augmented_image = to_numpy_volume(augmented["image"])
    augmented_label = to_numpy_volume(augmented["label"])
    augmented_slice = choose_slice_index(augmented_label)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    figure.suptitle(f"{item['patient']} {item['phase']} augmented sample {sample_idx}")
    metadata = f"slice={augmented_slice}"

    plot_overlay(
        axes[0],
        augmented_image[:, :, augmented_slice],
        augmented_label[:, :, augmented_slice],
        "Augmented image + label",
        metadata,
    )
    plot_image_only(
        axes[1],
        augmented_image[:, :, augmented_slice],
        "Augmented image only",
        metadata,
    )

    figure.tight_layout()

    output_path = output_dir / f"{item['patient']}_{item['phase']}_aug_{sample_idx}.png"
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output_path


def main() -> None:
    """Generate visual QC figures for a subset of cases."""
    args = parse_args()
    items = build_acdc_list()
    if not items:
        raise SystemExit("No ACDC items were found. Check config.py and dataset availability.")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    loader_transform = build_loader_transform()
    preprocess_transform = build_preprocess_transform(
        target_spacing=tuple(args.target_spacing),
        pad_size=tuple(args.pad_size),
    )
    train_transform = build_train_transform(
        target_spacing=tuple(args.target_spacing),
        patch_size=tuple(args.pad_size),
        num_samples=args.num_augmented_samples,
    )

    rng = random.Random(args.seed)
    selected_items = rng.sample(items, k=min(args.limit, len(items)))

    for item in selected_items:
        raw = loader_transform(item)
        processed = preprocess_transform(item)
        output_path = save_preprocess_figure(item, raw, processed, output_dir)
        print(f"Saved {output_path}")

        if args.include_augmented:
            augmented_samples = train_transform(item)
            for sample_idx, augmented in enumerate(augmented_samples, start=1):
                if sample_idx > args.num_augmented_samples:
                    break
                augmented_output = save_augmented_figure(item, augmented, sample_idx, output_dir)
                print(f"Saved {augmented_output}")


if __name__ == "__main__":
    main()
