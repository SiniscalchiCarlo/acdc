from __future__ import annotations

import json
import sys
import time
from typing import Any
from pathlib import Path

import numpy as np
import torch
from monai.data import decollate_batch, pad_list_data_collate
from monai.losses import DiceCELoss, DiceFocalLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.networks.nets import UNet, AttentionUnet, SegResNet
from monai.transforms import AsDiscrete, Compose, EnsureType
from tqdm import tqdm

# Allow the script to be executed from the repository root without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import training_config as cfg
from src.load_data_2D import (
    build_loaders,
    build_preprocessed_2d_list,
    load_preprocessed_2d_manifest,
    split_by_patient,
)
from src.transforms_2D import (
    build_preprocessed_train_transform,
    build_preprocessed_val_transform,
)

CLASS_NAMES = ("rv", "myo", "lv")


def init_wandb_run(config) -> Any | None:
    """Initialize a Weights & Biases run when enabled in config."""
    if not getattr(config, "WANDB_ENABLED", False):
        return None

    try:
        import wandb
    except ImportError:
        print("W&B logging requested but package 'wandb' is not installed. Continuing without W&B.")
        return None

    run_name = getattr(config, "WANDB_RUN_NAME", None)
    if isinstance(run_name, str) and not run_name.strip():
        run_name = None

    requested_mode = str(getattr(config, "WANDB_MODE", "online")).strip().lower()
    if requested_mode not in {"online", "offline", "disabled"}:
        print(f"Unknown WANDB_MODE '{requested_mode}', defaulting to 'online'.")
        requested_mode = "online"

    effective_mode = requested_mode
    login_timeout = int(getattr(config, "WANDB_LOGIN_TIMEOUT", 20))
    init_timeout = int(getattr(config, "WANDB_INIT_TIMEOUT", 30))
    fallback_to_offline = bool(getattr(config, "WANDB_FALLBACK_TO_OFFLINE", True))

    if requested_mode == "online":
        try:
            wandb.login(timeout=login_timeout)
        except Exception as ex:
            if fallback_to_offline:
                effective_mode = "offline"
                print(
                    f"W&B login failed/timed out ({ex}). "
                    f"Falling back to offline mode for this run."
                )
            else:
                print(f"W&B login failed/timed out ({ex}). Continuing without W&B.")
                return None

    try:
        run = wandb.init(
            project=config.WANDB_PROJECT,
            entity=getattr(config, "WANDB_ENTITY", None),
            name=run_name,
            mode=effective_mode,
            dir=str(REPO_ROOT),
            settings=wandb.Settings(init_timeout=init_timeout),
            config={
                "seed": config.SEED,
                "fold": config.FOLD,
                "n_splits": config.N_SPLITS,
                "model": config.MODEL,
                "epochs": config.EPOCHS,
                "batch_size": config.BATCH_SIZE,
                "lr": config.LR,
                "weight_decay": config.WEIGHT_DECAY,
                "lambda_dice": config.LAMBDA_DICE,
                "lambda_ce": config.LAMBDA_CE,
                "loss_function": config.LOSS_FUNCTION,
                "dynamic_class_weights": config.DYNAMIC_CLASS_WEIGHTS,
                "target_spacing": list(config.TARGET_SPACING),
                "patch_size": list(config.PATCH_SIZE),
                "preprocessed_root": None if config.PREPROCESSED_ROOT is None else str(config.PREPROCESSED_ROOT),
                "wandb_mode_effective": effective_mode,
            },
        )
        print(f"W&B run initialized in '{effective_mode}' mode: {run.name}")
        return run
    except Exception as ex:
        print(f"Failed to initialize W&B run: {ex}. Continuing without W&B.")
        return None


def log_wandb_epoch(
    wandb_run: Any | None,
    epoch: int,
    train_loss: float,
    learning_rate: float,
    epoch_duration_sec: float,
    samples_per_sec: float,
    max_gpu_memory_mb: float | None,
    val_metrics: dict[str, float | list[float]] | None,
    class_weights: torch.Tensor | None = None,
) -> None:
    """Log epoch metrics to Weights & Biases if a run is active."""
    if wandb_run is None:
        return

    payload: dict[str, float | int] = {
        "epoch": epoch,
        "train/loss": float(train_loss),
        "train/lr": float(learning_rate),
        "perf/epoch_duration_sec": float(epoch_duration_sec),
        "perf/train_samples_per_sec": float(samples_per_sec),
    }
    if max_gpu_memory_mb is not None:
        payload["perf/max_gpu_memory_mb"] = float(max_gpu_memory_mb)

    if val_metrics is not None:
        payload["val/loss"] = float(val_metrics["val_loss"])
        payload["val/dice"] = float(val_metrics["val_dice"])
        payload["val/hd95"] = float(val_metrics["val_hd95"])
        class_dice_scores = list(val_metrics["val_dice_per_class"])
        class_hd95_scores = list(val_metrics["val_hd95_per_class"])
        for idx, class_name in enumerate(CLASS_NAMES):
            if idx < len(class_dice_scores):
                payload[f"val/dice_{class_name}"] = float(class_dice_scores[idx])
            if idx < len(class_hd95_scores):
                payload[f"val/hd95_{class_name}"] = float(class_hd95_scores[idx])

    # Log dynamic class weights if provided — index +1 to skip background
    if class_weights is not None:
        for idx, class_name in enumerate(CLASS_NAMES):
            payload[f"class_weight/{class_name}"] = float(class_weights[idx + 1])

    wandb_run.log(payload, step=epoch)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_device_info(device: torch.device) -> None:
    """Print information about the device used for training."""
    print(f"Using device: {device}")

    if device.type == "cuda" and torch.cuda.is_available():
        try:
            print(f"CUDA available: True")
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            print(f"CUDA total memory (MB): {props.total_memory / (1024**2):.1f}")
            print(f"PyTorch CUDA version: {torch.version.cuda}")
            print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
        except Exception as ex:
            print(f"Could not query full CUDA details: {ex}")
    else:
        print("CUDA available: False (running on CPU)")


def get_model(config):
    model_name = config.MODEL

    if model_name == 'UNET':
        return build_model()
    elif model_name == 'ATTUNET':
        return build_model_attention()
    elif model_name == 'SEGRESNET':
        return build_model_residual()
    elif model_name == '25DATTUNET':
        return build_model_attention25D()
    else:
        available = ["UNET", "ATTUNET", "SEGRESNET"]
        raise ValueError(f"Invalid MODEL '{model_name}'. Choose from {available}")


def build_model() -> UNet:
    """Construct a modest 2D UNet for baseline slice-wise segmentation with residual units."""
    return UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=4,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )


def build_model_attention() -> AttentionUnet:
    """Construct Attention UNet."""
    return AttentionUnet(
        spatial_dims=2, # Still 2D UNet, treating each slice independently. Could experiment with 3D attention or stacking multiple slices as input channels in the future.
        in_channels=1, # Treating each slice independently, maybe try stack 3 slices as input channels
        out_channels=4,
        channels=(16, 32, 64, 128, 256),
        #channels=(32, 64, 128, 256, 512), # Try larger model? maybe not the best option
        strides=(2, 2, 2, 2),
    )

def build_model_attention25D() -> AttentionUnet:
    """Construct 2.5D Attention UNet."""
    return AttentionUnet(
        spatial_dims=2, # Still 2D UNet, treating each slice independently. Could experiment with 3D attention or stacking multiple slices as input channels in the future.
        in_channels=3, # Stack 3 slices as input channels
        out_channels=4,
        channels=(16, 32, 64, 128, 256),
        #channels=(32, 64, 128, 256, 512), # Try larger model? maybe not the best option
        strides=(2, 2, 2, 2),
    )


def build_model_residual() -> SegResNet:
    """Construct top-end SegResNet."""
    return SegResNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=4,
        init_filters=16,
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
    )


def build_post_transforms() -> tuple[Compose, Compose]:
    """Build post-processing transforms for predictions and labels before Dice."""
    post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=4)])
    post_label = Compose([EnsureType(), AsDiscrete(to_onehot=4)])
    return post_pred, post_label


def compute_class_weights(val_dice_per_class: list[float], device: torch.device) -> torch.Tensor:
    """Compute class weights inversely proportional to per-class Dice scores.

    Background gets weight 1.0. Classes with lower Dice get higher weight
    so the loss focuses more on them. Foreground weights are normalized
    so they sum to the number of foreground classes.
    """
    dice_scores = torch.tensor(val_dice_per_class, dtype=torch.float32)
    # Clamp to avoid division by zero if Dice is 1.0
    weights = 1.0 - dice_scores.clamp(0.0, 0.999)
    # Normalize so foreground weights sum to number of classes
    weights = weights / weights.sum() * len(dice_scores)
    # Prepend background weight of 1.0
    background_weight = torch.ones(1, dtype=torch.float32)
    return torch.cat([background_weight, weights], dim=0).to(device)

# Try CenterlineDiceLoss or Inter-Slice Centroid Smoothness Loss? 
# Problem with inter-slice smoothness = shuffle=True so random batches often do not contain many truly consecutive slices.
def build_loss_fn(class_weights: torch.Tensor) -> torch.nn.Module:
    """Build the loss function selected in config with the given class weights.

    Supports 'DiceCE' and 'DiceFocal'. Raises ValueError for unknown options.
    """
    loss_name = getattr(cfg, "LOSS_FUNCTION", "DiceCE").strip()

    if loss_name == "DiceCE":
        return DiceCELoss(
            to_onehot_y=True,
            softmax=True,
            lambda_dice=cfg.LAMBDA_DICE,
            lambda_ce=cfg.LAMBDA_CE,
            weight=class_weights,
        )
    elif loss_name == "DiceFocal":
        return DiceFocalLoss(
            to_onehot_y=True,
            softmax=True,
            lambda_dice=cfg.LAMBDA_DICE,
            lambda_focal=cfg.LAMBDA_CE,  # reuse LAMBDA_CE as focal weight
            weight=class_weights,
        )
    else:
        available = ["DiceCE", "DiceFocal"]
        raise ValueError(f"Invalid LOSS_FUNCTION '{loss_name}'. Choose from {available}")


def validate_preprocessed_manifest(manifest: dict[str, object]) -> None:
    """Fail fast when the saved offline preprocessing does not match the training config."""
    saved_config = manifest.get("config")
    if not isinstance(saved_config, dict):
        raise RuntimeError("Malformed preprocessed manifest: missing 'config' section.")

    expected = {
        "target_spacing": list(cfg.TARGET_SPACING),
        "patch_size": list(cfg.PATCH_SIZE),
        "include_background_slices": cfg.INCLUDE_BACKGROUND_SLICES,
        "min_label_pixels": cfg.MIN_LABEL_PIXELS,
    }
    mismatches: list[str] = []
    for key, expected_value in expected.items():
        actual_value = saved_config.get(key)
        if actual_value != expected_value:
            mismatches.append(f"{key}: expected {expected_value}, found {actual_value}")

    if mismatches:
        details = "; ".join(mismatches)
        raise RuntimeError(
            "Preprocessed dataset config does not match train_baseline_2d_config.py. "
            f"Regenerate the offline dataset or align the config. {details}"
        )


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

    for batch_idx, batch in enumerate(tqdm(loader, desc="Train", leave=False), start=1):
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
    """Run slice-wise validation and return loss, mean Dice, HD95, and per-class scores."""
    model.eval()
    mean_dice_metric = DiceMetric(include_background=False, reduction="mean")
    class_dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")
    class_hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean_batch")
    post_pred, post_label = build_post_transforms()
    losses: list[float] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Val", leave=False), start=1):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            logits = model(images)
            losses.append(float(loss_fn(logits, labels).item()))

            pred_list = [post_pred(x) for x in decollate_batch(logits)]
            label_list = [post_label(x) for x in decollate_batch(labels)]
            mean_dice_metric(y_pred=pred_list, y=label_list)
            class_dice_metric(y_pred=pred_list, y=label_list)
            hd95_metric(y_pred=pred_list, y=label_list)
            class_hd95_metric(y_pred=pred_list, y=label_list)

            if max_batches is not None and batch_idx >= max_batches:
                break

    mean_dice = float(mean_dice_metric.aggregate().item())
    class_dice_scores = class_dice_metric.aggregate().detach().cpu().numpy().astype(float).tolist()
    mean_hd95 = float(hd95_metric.aggregate().item())
    class_hd95_scores = class_hd95_metric.aggregate().detach().cpu().numpy().astype(float).tolist()

    mean_dice_metric.reset()
    class_dice_metric.reset()
    hd95_metric.reset()
    class_hd95_metric.reset()

    mean_loss = float(np.mean(losses)) if losses else float("nan")
    return {
        "val_loss": mean_loss,
        "val_dice": mean_dice,
        "val_dice_per_class": class_dice_scores,
        "val_hd95": mean_hd95,
        "val_hd95_per_class": class_hd95_scores,
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
    print_device_info(device)
    wandb_run = init_wandb_run(cfg)
    torch.manual_seed(cfg.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.SEED)
        torch.cuda.reset_peak_memory_stats()

    if cfg.PREPROCESSED_ROOT is None:
        raise RuntimeError("PREPROCESSED_ROOT must point to a generated preprocessed 2D dataset.")

    manifest = load_preprocessed_2d_manifest(cfg.PREPROCESSED_ROOT)
    validate_preprocessed_manifest(manifest)
    items = build_preprocessed_2d_list(cfg.PREPROCESSED_ROOT)
    train_items, val_items = split_by_patient(items, val_size=cfg.VAL_SIZE, seed=cfg.SEED)

    train_transform = build_preprocessed_train_transform(
        patch_size=cfg.PATCH_SIZE,
    )
    val_transform = build_preprocessed_val_transform(
        patch_size=cfg.PATCH_SIZE,
    )
    collate_fn = pad_list_data_collate

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
        collate_fn=collate_fn,
    )

    model = get_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=cfg.SCHEDULER_FACTOR,
        patience=cfg.SCHEDULER_PATIENCE,
        min_lr=cfg.SCHEDULER_MIN_LR,
    )

    # Start with equal weights for all 4 classes (background, RV, MYO, LV)
    class_weights = torch.ones(4, dtype=torch.float32).to(device)
    loss_fn = build_loss_fn(class_weights)

    # Log startup info
    print(f"Loss function: {cfg.LOSS_FUNCTION}")
    print(f"Dynamic class weighting: {'enabled' if cfg.DYNAMIC_CLASS_WEIGHTS else 'disabled'}")

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

            # Dynamically update class weights based on per-class Dice — only if enabled
            if cfg.DYNAMIC_CLASS_WEIGHTS:
                class_weights = compute_class_weights(val_metrics["val_dice_per_class"], device)
                loss_fn = build_loss_fn(class_weights)
                print(
                    f"Updated class weights — "
                    f"RV: {class_weights[1]:.3f} "
                    f"MYO: {class_weights[2]:.3f} "
                    f"LV: {class_weights[3]:.3f}"
                )

            if float(val_metrics["val_dice"]) > best_val_dice:
                best_val_dice = float(val_metrics["val_dice"])
                best_epoch = epoch
                best_metrics = {
                    "train_loss": round(train_loss, 6),
                    "val_loss": round(float(val_metrics["val_loss"]), 6),
                    "val_dice": round(float(val_metrics["val_dice"]), 6),
                    "val_dice_per_class": [round(float(score), 6) for score in list(val_metrics["val_dice_per_class"])],
                    "val_hd95": round(float(val_metrics["val_hd95"]), 4),
                    "val_hd95_per_class": [round(float(s), 4) for s in val_metrics["val_hd95_per_class"]],
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
                f"val_hd95={float(val_metrics['val_hd95']):.4f} "
                f"lr={float(optimizer.param_groups[0]['lr']):.6f} "
                f"time={epoch_duration_sec:.2f}s "
                f"throughput={samples_per_sec:.2f} samples/s"
            )
            log_wandb_epoch(
                wandb_run=wandb_run,
                epoch=epoch,
                train_loss=train_loss,
                learning_rate=float(optimizer.param_groups[0]["lr"]),
                epoch_duration_sec=epoch_duration_sec,
                samples_per_sec=samples_per_sec,
                max_gpu_memory_mb=max_gpu_memory_mb,
                val_metrics=val_metrics,
                class_weights=class_weights if cfg.DYNAMIC_CLASS_WEIGHTS else None,
            )
        else:
            print(
                f"Epoch {epoch}: train_loss={train_loss:.4f} "
                f"lr={current_lr:.6f} "
                f"time={epoch_duration_sec:.2f}s "
                f"throughput={samples_per_sec:.2f} samples/s"
            )
            log_wandb_epoch(
                wandb_run=wandb_run,
                epoch=epoch,
                train_loss=train_loss,
                learning_rate=current_lr,
                epoch_duration_sec=epoch_duration_sec,
                samples_per_sec=samples_per_sec,
                max_gpu_memory_mb=max_gpu_memory_mb,
                val_metrics=None,
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
        "best_val_hd95": None if best_metrics is None else best_metrics["val_hd95"],
        "best_val_hd95_per_class": None if best_metrics is None else best_metrics["val_hd95_per_class"],
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

    if wandb_run is not None:
        wandb_run.summary["best_epoch"] = int(best_epoch)
        if best_metrics is not None:
            wandb_run.summary["best_val_dice"] = float(best_metrics["val_dice"])
            wandb_run.summary["best_val_loss"] = float(best_metrics["val_loss"])
            wandb_run.summary["best_train_loss"] = float(best_metrics["train_loss"])
            wandb_run.summary["best_val_hd95"] = float(best_metrics["val_hd95"])
        wandb_run.summary["model_path"] = str(cfg.MODEL_OUTPUT)

        try:
            import wandb

            artifact = wandb.Artifact(
                name=f"model-{cfg.MODEL.lower()}",
                type="model",
                description=f"Best {cfg.MODEL} checkpoint from run {wandb_run.name}",
                metadata=best_metrics,
            )
            artifact.add_file(str(cfg.MODEL_OUTPUT))
            wandb_run.log_artifact(artifact)
        except Exception as ex:
            print(f"W&B artifact upload skipped: {ex}")

        wandb_run.finish()
        print("W&B run finished.")


if __name__ == "__main__":
    main()
