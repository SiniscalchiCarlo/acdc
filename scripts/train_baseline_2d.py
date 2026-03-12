from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
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

from src.load_data_2D import build_acdc_list, build_loaders, build_preprocessed_2d_list, split_by_patient
from src.transforms_2D import (
    build_preprocessed_train_transform,
    build_preprocessed_val_transform,
    build_train_transform,
    build_val_transform,
)

CLASS_NAMES = ("rv", "myo", "lv")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the 2D UNet baseline."""
    parser = argparse.ArgumentParser(
        description="Train a reproducible 2D UNet baseline on slice-wise ACDC segmentation."
    )
    parser.add_argument("--fold", type=int, default=0, help="GroupKFold validation fold.")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of patient-wise folds.")
    parser.add_argument("--epochs", type=int, default=50, help="Maximum number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Training and validation batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Adam weight decay.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
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
        help="Include slices without foreground labels in the dataset.",
    )
    parser.add_argument(
        "--min-label-pixels",
        type=int,
        default=1,
        help="Minimum foreground pixels for a slice to be marked as foreground.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers. Default 0 keeps the baseline portable and easier to debug.",
    )
    parser.add_argument(
        "--cache-rate-train",
        type=float,
        default=0.0,
        help="Training cache ratio. Increase when the preprocessing is stable and RAM allows it.",
    )
    parser.add_argument(
        "--cache-rate-val",
        type=float,
        default=0.0,
        help="Validation cache ratio. Increase when the preprocessing is stable and RAM allows it.",
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
    parser.add_argument("--val-every", type=int, default=1, help="Run validation every N epochs.")
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=10,
        help="Stop after this many validation checks without improvement. Use 0 to disable.",
    )
    parser.add_argument(
        "--scheduler-patience",
        type=int,
        default=4,
        help="ReduceLROnPlateau patience in validation checks.",
    )
    parser.add_argument(
        "--scheduler-factor",
        type=float,
        default=0.5,
        help="ReduceLROnPlateau multiplicative decay factor.",
    )
    parser.add_argument(
        "--scheduler-min-lr",
        type=float,
        default=1e-6,
        help="Lower bound for the learning rate scheduler.",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Optional cap for a quick smoke baseline before running a full fold.",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=None,
        help="Optional cap for faster validation during debugging.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "artifacts" / "baseline_metrics_2d.json",
        help="Where to write the baseline metrics summary.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "checkpoints_2d",
        help="Directory where best checkpoints will be saved.",
    )
    parser.add_argument(
        "--preprocessed-root",
        type=Path,
        default=None,
        help="Optional root directory of an offline-preprocessed 2D dataset manifest.",
    )
    return parser.parse_args()


def get_device() -> torch.device:
    """Use CUDA when available; otherwise fall back to CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model() -> UNet:
    """Construct a modest 2D UNet for baseline slice-wise segmentation."""
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


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device,
    max_batches: int | None,
) -> tuple[float, int]:
    """Run one training epoch and return the mean loss and number of seen samples."""
    model.train()
    running_loss = 0.0
    num_steps = 0
    num_samples = 0

    for batch_idx, batch in enumerate(loader, start=1):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        num_samples += int(images.shape[0])

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item())
        num_steps += 1

        if max_batches is not None and batch_idx >= max_batches:
            break

    return running_loss / max(num_steps, 1), num_samples


def validate(
    model: torch.nn.Module,
    loader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    max_batches: int | None,
) -> dict[str, float | list[float]]:
    """Run slice-wise validation and return loss, mean Dice, and per-class Dice."""
    model.eval()
    mean_dice_metric = DiceMetric(include_background=False, reduction="mean")
    class_dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
    post_pred, post_label = build_post_transforms()
    losses: list[float] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            logits = model(images)
            losses.append(float(loss_fn(logits, labels).item()))

            pred_list = [post_pred(x) for x in decollate_batch(logits)]
            label_list = [post_label(x) for x in decollate_batch(labels)]
            mean_dice_metric(y_pred=pred_list, y=label_list)
            class_dice_metric(y_pred=pred_list, y=label_list)

            if max_batches is not None and batch_idx >= max_batches:
                break

    mean_dice = float(mean_dice_metric.aggregate().item())
    class_scores = class_dice_metric.aggregate().detach().cpu().numpy().astype(float).tolist()
    mean_dice_metric.reset()
    class_dice_metric.reset()
    mean_loss = float(np.mean(losses)) if losses else float("nan")
    return {
        "val_loss": mean_loss,
        "val_dice": mean_dice,
        "val_dice_per_class": class_scores,
    }


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float | list[float]],
    checkpoint_path: Path,
) -> None:
    """Persist the best model checkpoint with enough state to resume inspection."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        checkpoint_path,
    )


def main() -> None:
    """Train and validate a reproducible 2D baseline suitable for benchmarking."""
    args = parse_args()
    device = get_device()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()

    if args.preprocessed_root is None:
        items = build_acdc_list(
            include_background_slices=args.include_background_slices,
            min_label_pixels=args.min_label_pixels,
        )
    else:
        items = build_preprocessed_2d_list(args.preprocessed_root)
    train_items, val_items = split_by_patient(items, n_splits=args.n_splits, fold=args.fold)

    if args.preprocessed_root is None:
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
    else:
        train_transform = build_preprocessed_train_transform(
            patch_size=tuple(args.patch_size),
        )
        val_transform = build_preprocessed_val_transform(
            patch_size=tuple(args.patch_size),
        )

    train_loader, val_loader = build_loaders(
        train_items=train_items,
        val_items=val_items,
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
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        min_lr=args.scheduler_min_lr,
    )
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)

    history: list[dict[str, float | int | list[float]]] = []
    best_val_dice = -1.0
    best_epoch = 0
    best_metrics: dict[str, float | list[float]] | None = None
    epochs_without_improvement = 0
    checkpoint_path = args.checkpoint_dir / f"baseline_2d_fold{args.fold}_best.pt"

    print(f"Device: {device}")
    print(f"Train slices: {len(train_items)} | Val slices: {len(val_items)}")

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        train_loss, train_samples = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            max_batches=args.max_train_batches,
        )
        epoch_duration_sec = time.perf_counter() - epoch_start_time
        samples_per_sec = train_samples / max(epoch_duration_sec, 1e-8)
        max_gpu_memory_mb = None
        if torch.cuda.is_available():
            max_gpu_memory_mb = torch.cuda.max_memory_allocated(device=device) / (1024**2)

        current_lr = float(optimizer.param_groups[0]["lr"])
        epoch_metrics: dict[str, float | int | list[float]] = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "lr": round(current_lr, 8),
            "epoch_duration_sec": round(epoch_duration_sec, 4),
            "train_samples": train_samples,
            "train_samples_per_sec": round(samples_per_sec, 4),
        }
        if max_gpu_memory_mb is not None:
            epoch_metrics["max_gpu_memory_mb"] = round(float(max_gpu_memory_mb), 2)

        if epoch % args.val_every == 0:
            val_metrics = validate(
                model=model,
                loader=val_loader,
                loss_fn=loss_fn,
                device=device,
                max_batches=args.max_val_batches,
            )
            scheduler.step(float(val_metrics["val_dice"]))

            epoch_metrics["val_loss"] = round(float(val_metrics["val_loss"]), 6)
            epoch_metrics["val_dice"] = round(float(val_metrics["val_dice"]), 6)
            epoch_metrics["val_dice_per_class"] = [
                round(float(score), 6) for score in list(val_metrics["val_dice_per_class"])
            ]

            if float(val_metrics["val_dice"]) > best_val_dice:
                best_val_dice = float(val_metrics["val_dice"])
                best_epoch = epoch
                best_metrics = {
                    "val_loss": round(float(val_metrics["val_loss"]), 6),
                    "val_dice": round(float(val_metrics["val_dice"]), 6),
                    "val_dice_per_class": epoch_metrics["val_dice_per_class"],
                }
                epochs_without_improvement = 0
                save_checkpoint(model, optimizer, epoch, best_metrics, checkpoint_path)
            else:
                epochs_without_improvement += 1

            print(
                f"Epoch {epoch}: train_loss={train_loss:.4f} "
                f"val_loss={float(val_metrics['val_loss']):.4f} "
                f"val_dice={float(val_metrics['val_dice']):.4f} "
                f"lr={float(optimizer.param_groups[0]['lr']):.6f} "
                f"time={epoch_duration_sec:.2f}s "
                f"throughput={samples_per_sec:.2f} samples/s"
            )
        else:
            print(
                f"Epoch {epoch}: train_loss={train_loss:.4f} "
                f"lr={current_lr:.6f} "
                f"time={epoch_duration_sec:.2f}s "
                f"throughput={samples_per_sec:.2f} samples/s"
            )

        history.append(epoch_metrics)

        if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
            print(f"Early stopping at epoch {epoch} after {epochs_without_improvement} stale validations.")
            break

    summary = {
        "config": {
            "fold": args.fold,
            "n_splits": args.n_splits,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "patch_size": list(args.patch_size),
            "target_spacing": list(args.target_spacing),
            "foreground_margin": args.foreground_margin,
            "include_background_slices": args.include_background_slices,
            "min_label_pixels": args.min_label_pixels,
            "num_workers": args.num_workers,
            "cache_rate_train": args.cache_rate_train,
            "cache_rate_val": args.cache_rate_val,
            "pin_memory": args.pin_memory,
            "val_every": args.val_every,
            "early_stop_patience": args.early_stop_patience,
            "scheduler_patience": args.scheduler_patience,
            "scheduler_factor": args.scheduler_factor,
            "scheduler_min_lr": args.scheduler_min_lr,
            "max_train_batches": args.max_train_batches,
            "max_val_batches": args.max_val_batches,
            "device": str(device),
            "preprocessed_root": None if args.preprocessed_root is None else str(args.preprocessed_root),
        },
        "train_cases": len(train_items),
        "val_cases": len(val_items),
        "best_epoch": best_epoch,
        "best_val_dice": None if best_metrics is None else best_metrics["val_dice"],
        "best_val_loss": None if best_metrics is None else best_metrics["val_loss"],
        "best_val_dice_per_class": None if best_metrics is None else best_metrics["val_dice_per_class"],
        "class_names": list(CLASS_NAMES),
        "checkpoint_path": str(checkpoint_path) if best_metrics is not None else None,
        "history": history,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(f"Saved metrics to: {args.output}")
    if best_metrics is not None:
        print(f"Saved best checkpoint to: {checkpoint_path}")


if __name__ == "__main__":
    main()
