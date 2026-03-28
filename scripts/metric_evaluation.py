import sys
import torch
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import training_config as cfg
from training import get_device, get_model, validate
from src.load_data_2D import (
    build_preprocessed_2d_list,
    load_preprocessed_2d_manifest,
    split_by_patient,
    build_loaders,
)
from src.transforms_2D import build_preprocessed_val_transform
from monai.data import pad_list_data_collate
from monai.losses import DiceCELoss

# -----------------------------
# Settings — change these to evaluate different models
# -----------------------------
MODELS_TO_EVALUATE = [
    ("Baseline UNET",      "UNET",      Path("artifacts") / "models" / "Baseline_UNET.pt"),
    ("Baseline ATTUNET",   "ATTUNET",   Path("artifacts") / "models" / "Baseline_ATTUNET.pt"),
    ("Baseline SEGRESNET", "SEGRESNET", Path("artifacts") / "models" / "Baseline_SEGRESNET.pt"),
]


def evaluate_model(name: str, model_type: str, model_path: Path, device: torch.device) -> dict:
    """Load a saved model and compute Dice + HD95 on the validation set."""
    print(f"\nEvaluating: {name}")

    # Temporarily override model type in config so get_model builds the right architecture
    original_model = cfg.MODEL
    cfg.MODEL = model_type

    # Load model with correct architecture
    model = get_model(cfg).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Restore original config
    cfg.MODEL = original_model

    # Load validation data
    manifest = load_preprocessed_2d_manifest(cfg.PREPROCESSED_ROOT)
    items = build_preprocessed_2d_list(cfg.PREPROCESSED_ROOT)
    _, val_items = split_by_patient(items, n_splits=cfg.N_SPLITS, fold=cfg.FOLD)
    val_transform = build_preprocessed_val_transform(patch_size=cfg.PATCH_SIZE)
    _, val_loader = build_loaders(
        train_items=val_items,
        val_items=val_items,
        train_transform=val_transform,
        val_transform=val_transform,
        batch_size=8,
        num_workers=2,
        cache_rate_train=0.0,
        cache_rate_val=1.0,
        seed=cfg.SEED,
        pin_memory=cfg.PIN_MEMORY,
        collate_fn=pad_list_data_collate,
    )

    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    metrics = validate(model, val_loader, loss_fn, device, max_batches=None)

    # Print results
    print(f"  val_dice:          {metrics['val_dice']:.4f}")
    print(f"  val_hd95:          {metrics['val_hd95']:.4f}")
    print(f"  val_dice_rv:       {metrics['val_dice_per_class'][0]:.4f}")
    print(f"  val_dice_myo:      {metrics['val_dice_per_class'][1]:.4f}")
    print(f"  val_dice_lv:       {metrics['val_dice_per_class'][2]:.4f}")
    print(f"  val_hd95_rv:       {metrics['val_hd95_per_class'][0]:.4f}")
    print(f"  val_hd95_myo:      {metrics['val_hd95_per_class'][1]:.4f}")
    print(f"  val_hd95_lv:       {metrics['val_hd95_per_class'][2]:.4f}")

    return metrics


def main() -> None:
    """Evaluate all saved models and print a comparison summary."""
    device = get_device()

    results = {}
    for name, model_type, path in MODELS_TO_EVALUATE:
        if not path.exists():
            print(f"Skipping {name} — file not found: {path}")
            continue
        results[name] = evaluate_model(name, model_type, path, device)

    # Print comparison summary
    print("\n--- Summary ---")
    print(f"{'Model':<25} {'Dice':>8} {'HD95':>8}")
    print("-" * 45)
    for name, model_type, path in MODELS_TO_EVALUATE:
        if name in results:
            print(f"{name:<25} {results[name]['val_dice']:>8.4f} {results[name]['val_hd95']:>8.4f}")


if __name__ == "__main__":
    main()
