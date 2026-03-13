from __future__ import annotations

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

import train_baseline_2d_config as cfg
from src.load_data_2D import build_acdc_list, build_loaders, build_preprocessed_2d_list, split_by_patient
from src.transforms_2D import (
    build_preprocessed_train_transform,
    build_preprocessed_val_transform,
    build_train_transform,
    build_val_transform,
)

CLASS_NAMES = ("rv", "myo", "lv")


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


def save_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float | list[float]],
    model_path: Path,
) -> None:
    """Persist the final trained model and minimal run state."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        model_path,
    )


def main() -> None:
    """Train and validate a reproducible 2D baseline using config-only settings."""
    device = get_device()
    torch.manual_seed(cfg.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.SEED)
        torch.cuda.reset_peak_memory_stats()

    if cfg.PREPROCESSED_ROOT is None:
        items = build_acdc_list(
            include_background_slices=cfg.INCLUDE_BACKGROUND_SLICES,
            min_label_pixels=cfg.MIN_LABEL_PIXELS,
        )
    else:
        items = build_preprocessed_2d_list(cfg.PREPROCESSED_ROOT)
    train_items, val_items = split_by_patient(items, n_splits=cfg.N_SPLITS, fold=cfg.FOLD)

    if cfg.PREPROCESSED_ROOT is None:
        train_transform = build_train_transform(
            target_spacing=cfg.TARGET_SPACING,
            patch_size=cfg.PATCH_SIZE,
            foreground_margin=cfg.FOREGROUND_MARGIN,
        )
        val_transform = build_val_transform(
            target_spacing=cfg.TARGET_SPACING,
            patch_size=cfg.PATCH_SIZE,
            foreground_margin=cfg.FOREGROUND_MARGIN,
        )
    else:
        train_transform = build_preprocessed_train_transform(
            patch_size=cfg.PATCH_SIZE,
        )
        val_transform = build_preprocessed_val_transform(
            patch_size=cfg.PATCH_SIZE,
        )

    train_loader, val_loader = build_loaders(
        train_items=train_items,
        val_items=val_items,
        train_transform=train_transform,
        val_transform=val_transform,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        cache_rate_train=cfg.CACHE_RATE_TRAIN,
        cache_rate_val=cfg.CACHE_RATE_VAL,
        seed=cfg.SEED,
        pin_memory=cfg.PIN_MEMORY,
    )

    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=cfg.SCHEDULER_FACTOR,
        patience=cfg.SCHEDULER_PATIENCE,
        min_lr=cfg.SCHEDULER_MIN_LR,
    )
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)

    best_val_dice = -1.0
    best_epoch = 0
    best_metrics: dict[str, float | list[float]] | None = None
    best_epoch_duration_sec = None
    best_epoch_throughput = None
    best_epoch_gpu_memory_mb = None
    epochs_without_improvement = 0
    epochs_run = 0

    print(f"Device: {device}")
    print(f"Train slices: {len(train_items)} | Val slices: {len(val_items)}")

    for epoch in range(1, cfg.EPOCHS + 1):
        epochs_run = epoch
        epoch_start_time = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        train_loss, train_samples = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            max_batches=cfg.MAX_TRAIN_BATCHES,
        )
        epoch_duration_sec = time.perf_counter() - epoch_start_time
        samples_per_sec = train_samples / max(epoch_duration_sec, 1e-8)
        max_gpu_memory_mb = None
        if torch.cuda.is_available():
            max_gpu_memory_mb = torch.cuda.max_memory_allocated(device=device) / (1024**2)

        current_lr = float(optimizer.param_groups[0]["lr"])

        if epoch % cfg.VAL_EVERY == 0:
            val_metrics = validate(
                model=model,
                loader=val_loader,
                loss_fn=loss_fn,
                device=device,
                max_batches=cfg.MAX_VAL_BATCHES,
            )
            scheduler.step(float(val_metrics["val_dice"]))

            if float(val_metrics["val_dice"]) > best_val_dice:
                best_val_dice = float(val_metrics["val_dice"])
                best_epoch = epoch
                best_metrics = {
                    "train_loss": round(train_loss, 6),
                    "val_loss": round(float(val_metrics["val_loss"]), 6),
                    "val_dice": round(float(val_metrics["val_dice"]), 6),
                    "val_dice_per_class": [round(float(score), 6) for score in list(val_metrics["val_dice_per_class"])],
                }
                best_epoch_duration_sec = round(epoch_duration_sec, 4)
                best_epoch_throughput = round(samples_per_sec, 4)
                best_epoch_gpu_memory_mb = None if max_gpu_memory_mb is None else round(float(max_gpu_memory_mb), 2)
                epochs_without_improvement = 0
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

        if cfg.EARLY_STOP_PATIENCE > 0 and epochs_without_improvement >= cfg.EARLY_STOP_PATIENCE:
            print(f"Early stopping at epoch {epoch} after {epochs_without_improvement} stale validations.")
            break

    save_model(
        model=model,
        optimizer=optimizer,
        epoch=epochs_run,
        metrics={} if best_metrics is None else best_metrics,
        model_path=cfg.MODEL_OUTPUT,
    )

    summary = {
        "fold": cfg.FOLD,
        "epochs_run": epochs_run,
        "train_slices": len(train_items),
        "val_slices": len(val_items),
        "best_epoch": best_epoch,
        "best_val_dice": None if best_metrics is None else best_metrics["val_dice"],
        "best_val_loss": None if best_metrics is None else best_metrics["val_loss"],
        "best_train_loss": None if best_metrics is None else best_metrics["train_loss"],
        "best_val_dice_per_class": None if best_metrics is None else best_metrics["val_dice_per_class"],
        "class_names": list(CLASS_NAMES),
        "best_epoch_duration_sec": best_epoch_duration_sec,
        "best_epoch_train_samples_per_sec": best_epoch_throughput,
        "best_epoch_max_gpu_memory_mb": best_epoch_gpu_memory_mb,
        "model_path": str(cfg.MODEL_OUTPUT),
        "preprocessed_root": None if cfg.PREPROCESSED_ROOT is None else str(cfg.PREPROCESSED_ROOT),
    }

    cfg.OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    cfg.OUTPUT.write_text(json.dumps(summary, indent=2))
    print(f"Saved metrics to: {cfg.OUTPUT}")
    print(f"Saved final model to: {cfg.MODEL_OUTPUT}")


if __name__ == "__main__":
    main()
