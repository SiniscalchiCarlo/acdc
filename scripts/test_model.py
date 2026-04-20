from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from monai.data import pad_list_data_collate
from monai.losses import DiceCELoss

# Allow the script to be executed from the repository root without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config as cfg
from scripts.training import CLASS_NAMES, get_device, get_model, print_device_info, validate
from src.load_data_2D import build_acdc_list, build_loaders
from src.mode_compatibility import expected_input_channels_for_model, validate_model_preprocessing_compatibility
from src.transforms_2D import build_preprocessing_transform


def validate_config() -> Path:
    """Validate required test config before any heavy work starts."""
    if not cfg.MODEL_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {cfg.MODEL_PATH}")
    if not cfg.MODEL_PATH.is_file():
        raise RuntimeError(f"MODEL_PATH must point to a model checkpoint file, got: {cfg.MODEL_PATH}")
    if cfg.TEST_ROOT is None:
        raise RuntimeError("TEST_PATH is not set. Export TEST_PATH to the raw test dataset directory.")
    if not cfg.TEST_ROOT.exists():
        raise FileNotFoundError(f"Test dataset directory not found: {cfg.TEST_ROOT}")
    if not cfg.TEST_ROOT.is_dir():
        raise RuntimeError(f"TEST_PATH must point to a directory, got: {cfg.TEST_ROOT}")
    return cfg.MODEL_PATH


def assert_input_channels_match(model: torch.nn.Module) -> None:
    """Fail fast when the configured architecture does not match the checkpoint weights."""
    expected_in_channels = expected_input_channels_for_model(cfg.MODEL)
    actual_in_channels = int(getattr(model, "in_channels", expected_in_channels))
    if actual_in_channels != expected_in_channels:
        raise RuntimeError(
            f"Model '{cfg.MODEL}' expects {expected_in_channels} input channel(s), "
            f"but the constructed network exposes {actual_in_channels}. "
            "Check MODEL in config.py."
        )


def print_test_run_summary(model_path: Path) -> None:
    """Print the most important evaluation configuration before heavy work starts."""
    expected_channels = expected_input_channels_for_model(cfg.MODEL)
    print("Test configuration:")
    print(f"  model: {cfg.MODEL}")
    print(f"  preprocessing_mode: {cfg.PREPROCESSING_MODE}")
    print(f"  expected_input_channels: {expected_channels}")
    print(f"  model_path: {model_path}")
    print(f"  test_root: {cfg.TEST_ROOT}")
    print(f"  patch_size: {cfg.PATCH_SIZE}")
    print(f"  target_spacing: {cfg.TARGET_SPACING}")
    print(f"  include_background_slices: {cfg.INCLUDE_BACKGROUND_SLICES}")
    print(f"  min_label_pixels: {cfg.MIN_LABEL_PIXELS}")
    print(f"  batch_size: {cfg.BATCH_SIZE}")
    print(f"  num_workers: {cfg.NUM_WORKERS}")
    print(f"  metrics_output: {cfg.OUTPUT}")


def main() -> None:
    """Evaluate one saved checkpoint on the raw test set."""
    model_path = validate_config()
    validate_model_preprocessing_compatibility(cfg.MODEL, cfg.PREPROCESSING_MODE)
    device = get_device()
    print_device_info(device)
    print_test_run_summary(model_path)

    torch.manual_seed(cfg.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.SEED)

    test_items = build_acdc_list(
        dataset_root=cfg.TEST_ROOT,
        include_background_slices=cfg.INCLUDE_BACKGROUND_SLICES,
        min_label_pixels=cfg.MIN_LABEL_PIXELS,
    )

    test_transform = build_preprocessing_transform(
        target_spacing=cfg.TARGET_SPACING,
        patch_size=cfg.PATCH_SIZE,
        preprocessing_mode=cfg.PREPROCESSING_MODE,
    )
    _, val_loader = build_loaders(
        train_items=test_items,
        val_items=test_items,
        train_transform=test_transform,
        val_transform=test_transform,
        batch_size=cfg.TEST_BATCH_SIZE,
        num_workers=cfg.TEST_NUM_WORKERS,
        cache_rate_train=0.0,
        cache_rate_val=cfg.TEST_CACHE_RATE_VAL,
        seed=cfg.SEED,
        pin_memory=cfg.TEST_PIN_MEMORY,
        collate_fn=pad_list_data_collate,
    )

    model = get_model(cfg).to(device)
    assert_input_channels_match(model)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    metrics = validate(
        model=model,
        loader=val_loader,
        loss_fn=loss_fn,
        device=device,
        max_batches=cfg.TEST_MAX_VAL_BATCHES,
    )

    summary = {
        "model": cfg.MODEL,
        "model_path": str(model_path),
        "test_root": str(cfg.TEST_ROOT),
        "test_slices": len(test_items),
        "test_loss": round(float(metrics["val_loss"]), 6),
        "test_dice": round(float(metrics["val_dice"]), 6),
        "test_dice_per_class": [round(float(score), 6) for score in metrics["val_dice_per_class"]],
        "test_hd95": round(float(metrics["val_hd95"]), 4),
        "test_hd95_per_class": [round(float(score), 4) for score in metrics["val_hd95_per_class"]],
        "class_names": list(CLASS_NAMES),
    }

    cfg.TEST_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    cfg.TEST_OUTPUT.write_text(json.dumps(summary, indent=2))

    print(f"Tested model: {model_path}")
    print(f"Test Dice: {summary['test_dice']:.4f}")
    print(f"Test HD95: {summary['test_hd95']:.4f}")
    print(f"Saved test metrics to: {cfg.TEST_OUTPUT}")


if __name__ == "__main__":
    main()
