from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from monai.data import decollate_batch
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import AsDiscrete, Compose, EnsureType

# Allow the script to be executed from the repository root without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.load_data_2D import build_acdc_list, build_loaders, split_by_patient
from src.transforms_2D import build_train_transform, build_val_transform


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the 2D training sanity check."""
    parser = argparse.ArgumentParser(
        description=(
            "Run a small 2D overfit sanity check. If the preprocessing is healthy, "
            "the model should fit a tiny subset quickly."
        )
    )
    parser.add_argument("--fold", type=int, default=0, help="GroupKFold validation fold.")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of patient-wise folds.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of sanity-check epochs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for the sanity run.")
    parser.add_argument("--subset-size", type=int, default=16, help="How many training slices to keep.")
    parser.add_argument("--val-subset-size", type=int, default=16, help="How many validation slices to keep.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(192, 192),
        metavar=("PX", "PY"),
        help="2D size passed to the train and validation preprocessing.",
    )
    parser.add_argument(
        "--target-spacing",
        type=float,
        nargs=3,
        default=(1.25, 1.25, -1.0),
        metavar=("SX", "SY", "SZ"),
        help="Spacing passed to the train and validation preprocessing.",
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
        help="Include background-only slices in the sanity-check pool.",
    )
    parser.add_argument(
        "--min-label-pixels",
        type=int,
        default=1,
        help="Minimum foreground pixels for a slice to be flagged as foreground.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers. Default 0 keeps the sanity check portable and easier to debug.",
    )
    parser.add_argument(
        "--cache-rate-train",
        type=float,
        default=0.0,
        help="Training cache ratio. Use 0.0 for a conservative sanity run.",
    )
    parser.add_argument(
        "--cache-rate-val",
        type=float,
        default=0.0,
        help="Validation cache ratio. Use 0.0 for a conservative sanity run.",
    )
    pin_memory_group = parser.add_mutually_exclusive_group()
    pin_memory_group.add_argument(
        "--pin-memory",
        dest="pin_memory",
        action="store_true",
        help="Enable DataLoader pinned-memory staging for faster CPU-to-GPU transfers.",
    )
    pin_memory_group.add_argument(
        "--no-pin-memory",
        dest="pin_memory",
        action="store_false",
        help="Disable DataLoader pinned-memory staging. Useful when multiprocessing is unstable.",
    )
    parser.set_defaults(pin_memory=True)
    parser.add_argument(
        "--target-train-dice",
        type=float,
        default=0.85,
        help="Expected train Dice threshold for the tiny-subset overfit check.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "artifacts" / "train_sanity_2d.json",
        help="Where to write the sanity-check metrics summary.",
    )
    return parser.parse_args()


def get_device() -> torch.device:
    """Use CUDA when available; otherwise fall back to CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model() -> UNet:
    """Construct a modest 2D UNet for the sanity check."""
    return UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=4,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )


def build_post_transforms() -> tuple[Compose, Compose]:
    """Build post-processing transforms for predictions and labels before Dice."""
    post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=4)])
    post_label = Compose([EnsureType(), AsDiscrete(to_onehot=4)])
    return post_pred, post_label


def evaluate(model: torch.nn.Module, loader, device: torch.device) -> tuple[float, float]:
    """Return mean loss proxy via Dice metric plus exact mean Dice on a loader."""
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    post_pred, post_label = build_post_transforms()
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)

    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            logits = model(images)
            losses.append(float(loss_fn(logits, labels).item()))

            pred_list = [post_pred(x) for x in decollate_batch(logits)]
            label_list = [post_label(x) for x in decollate_batch(labels)]
            dice_metric(y_pred=pred_list, y=label_list)

    dice = float(dice_metric.aggregate().item())
    dice_metric.reset()
    mean_loss = float(sum(losses) / max(len(losses), 1))
    return mean_loss, dice


def sample_subset(items: list[dict[str, object]], subset_size: int, seed: int) -> list[dict[str, object]]:
    """Sample a stable subset while preferring foreground slices when possible."""
    rng = random.Random(seed)
    foreground_items = [item for item in items if bool(item["has_foreground"])]
    background_items = [item for item in items if not bool(item["has_foreground"])]

    if subset_size >= len(items):
        return list(items)

    chosen: list[dict[str, object]] = []
    if foreground_items:
        num_foreground = min(len(foreground_items), subset_size)
        chosen.extend(rng.sample(foreground_items, k=num_foreground))

    remaining = subset_size - len(chosen)
    if remaining > 0 and background_items:
        chosen.extend(rng.sample(background_items, k=min(len(background_items), remaining)))

    if len(chosen) < subset_size:
        remaining_pool = [item for item in items if item not in chosen]
        chosen.extend(rng.sample(remaining_pool, k=subset_size - len(chosen)))

    rng.shuffle(chosen)
    return chosen


def main() -> None:
    """Run a tiny-subset overfit check and save the metrics summary."""
    args = parse_args()
    device = get_device()
    torch.manual_seed(args.seed)

    items = build_acdc_list(
        include_background_slices=args.include_background_slices,
        min_label_pixels=args.min_label_pixels,
    )
    train_items, val_items = split_by_patient(items, n_splits=args.n_splits, fold=args.fold)
    train_subset = sample_subset(train_items, subset_size=args.subset_size, seed=args.seed)
    val_subset = sample_subset(val_items, subset_size=args.val_subset_size, seed=args.seed + 1)

    train_transform = build_train_transform(
        target_spacing=tuple(args.target_spacing),
        patch_size=tuple(args.patch_size),
        foreground_margin=args.foreground_margin,
    )
    val_transform = build_val_transform(
        target_spacing=tuple(args.target_spacing),
        patch_size=tuple(args.patch_size),
        foreground_margin=args.foreground_margin,
    )

    train_loader, val_loader = build_loaders(
        train_items=train_subset,
        val_items=val_subset,
        train_transform=train_transform,
        val_transform=val_transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_rate_train=args.cache_rate_train,
        cache_rate_val=args.cache_rate_val,
        seed=args.seed,
        pin_memory=args.pin_memory,
    )

    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)

    initial_train_loss, initial_train_dice = evaluate(model, train_loader, device)
    initial_val_loss, initial_val_dice = evaluate(model, val_loader, device)

    history: list[dict[str, float | int]] = []
    best_train_dice = initial_train_dice

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        num_steps = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            num_steps += 1

        mean_train_loss = running_loss / max(num_steps, 1)
        eval_train_loss, eval_train_dice = evaluate(model, train_loader, device)
        eval_val_loss, eval_val_dice = evaluate(model, val_loader, device)
        best_train_dice = max(best_train_dice, eval_train_dice)

        history.append(
            {
                "epoch": epoch,
                "train_loss_step_mean": round(mean_train_loss, 6),
                "train_loss_eval": round(eval_train_loss, 6),
                "train_dice": round(eval_train_dice, 6),
                "val_loss_eval": round(eval_val_loss, 6),
                "val_dice": round(eval_val_dice, 6),
            }
        )

        print(
            f"Epoch {epoch}: train_loss={mean_train_loss:.4f} "
            f"train_dice={eval_train_dice:.4f} val_dice={eval_val_dice:.4f}"
        )

    passed = best_train_dice >= args.target_train_dice
    summary = {
        "config": {
            "fold": args.fold,
            "n_splits": args.n_splits,
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "subset_size": args.subset_size,
            "val_subset_size": args.val_subset_size,
            "lr": args.lr,
            "patch_size": list(args.patch_size),
            "target_spacing": list(args.target_spacing),
            "foreground_margin": args.foreground_margin,
            "include_background_slices": args.include_background_slices,
            "min_label_pixels": args.min_label_pixels,
            "num_workers": args.num_workers,
            "cache_rate_train": args.cache_rate_train,
            "cache_rate_val": args.cache_rate_val,
            "pin_memory": args.pin_memory,
            "target_train_dice": args.target_train_dice,
            "device": str(device),
        },
        "train_subset_size": len(train_subset),
        "val_subset_size": len(val_subset),
        "initial_train_loss": round(initial_train_loss, 6),
        "initial_train_dice": round(initial_train_dice, 6),
        "initial_val_loss": round(initial_val_loss, 6),
        "initial_val_dice": round(initial_val_dice, 6),
        "best_train_dice": round(best_train_dice, 6),
        "passed": passed,
        "history": history,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))

    print(f"Best train Dice: {best_train_dice:.4f}")
    print(f"Target train Dice: {args.target_train_dice:.4f}")
    print(f"Sanity check passed: {passed}")
    print(f"Saved metrics to: {args.output}")


if __name__ == "__main__":
    main()
