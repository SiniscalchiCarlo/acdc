from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Allow the script to be executed from the repository root without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.load_data_2D import build_acdc_list
from src.transforms_2D import build_val_transform


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for offline 2D preprocessing."""
    parser = argparse.ArgumentParser(
        description="Precompute deterministic 2D ACDC preprocessing and save each slice to disk."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "preprocessed_2d",
        help="Directory where preprocessed slices and the manifest will be saved.",
    )
    parser.add_argument(
        "--target-spacing",
        type=float,
        nargs=3,
        default=(1.25, 1.25, -1.0),
        metavar=("SX", "SY", "SZ"),
        help="Spacing passed to deterministic 2D preprocessing.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(192, 192),
        metavar=("PX", "PY"),
        help="Final 2D size passed to deterministic preprocessing.",
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
        help="Include background-only slices in the preprocessed dataset.",
    )
    parser.add_argument(
        "--min-label-pixels",
        type=int,
        default=1,
        help="Minimum foreground pixels for a slice to be marked as foreground.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of slices to preprocess for smoke tests.",
    )
    return parser.parse_args()


def main() -> None:
    """Precompute deterministic preprocessing for all selected 2D slices."""
    args = parse_args()
    items = build_acdc_list(
        include_background_slices=args.include_background_slices,
        min_label_pixels=args.min_label_pixels,
    )
    if args.limit is not None:
        items = items[: args.limit]
    transform = build_val_transform(
        target_spacing=tuple(args.target_spacing),
        patch_size=tuple(args.patch_size),
        foreground_margin=args.foreground_margin,
    )

    output_dir = args.output_dir
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    manifest_items: list[dict[str, object]] = []
    for item in items:
        transformed = transform(item)
        image = np.asarray(transformed["image"])[0].astype(np.float32, copy=False)
        label = np.asarray(transformed["label"])[0].astype(np.int16, copy=False)

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
            "target_spacing": list(args.target_spacing),
            "patch_size": list(args.patch_size),
            "foreground_margin": args.foreground_margin,
            "include_background_slices": args.include_background_slices,
            "min_label_pixels": args.min_label_pixels,
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
