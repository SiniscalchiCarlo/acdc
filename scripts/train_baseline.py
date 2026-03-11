from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import AsDiscrete, Compose, EnsureType

# Allow the script to be executed from the repository root without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.load_data import build_acdc_list, build_loaders, split_by_patient
from src.transforms import build_train_transform, build_val_transform


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the baseline preprocessing-validation run."""
    parser = argparse.ArgumentParser(
        description=(
            "Train a minimal segmentation baseline so preprocessing choices can be judged "
            "against validation metrics instead of geometry checks alone."
        )
    )
    parser.add_argument("--fold", type=int, default=0, help="GroupKFold validation fold.")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of patient-wise folds.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=1, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=3,
        default=(192, 192, 16),
        metavar=("PX", "PY", "PZ"),
        help="Patch size used for training crops and validation sliding-window inference.",
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
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers. Default 0 keeps the baseline portable and easier to debug.",
    )
    parser.add_argument(
        "--cache-rate-train",
        type=float,
        default=0.0,
        help="Training cache ratio. Start at 0.0 to reduce memory variables in the baseline.",
    )
    parser.add_argument(
        "--cache-rate-val",
        type=float,
        default=0.0,
        help="Validation cache ratio. Keep 0.0 for the same reason as training.",
    )
    parser.add_argument(
        "--val-every",
        type=int,
        default=1,
        help="Run validation every N epochs.",
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
        default=REPO_ROOT / "artifacts" / "baseline_metrics.json",
        help="Where to write the baseline metrics summary.",
    )
    return parser.parse_args()


def get_device() -> torch.device:
    """Pick CUDA when available because 3D segmentation is expensive on CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model() -> UNet:
    """Construct a modest 3D UNet to test the preprocessing without architecture tuning."""
    # This network is intentionally small: the goal is not to maximize challenge performance
    # yet, but to establish whether the current preprocessing supports stable learning.
    return UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=4,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
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
) -> float:
    """Run one training epoch and return the mean loss."""
    model.train()
    running_loss = 0.0
    num_steps = 0

    for batch_idx, batch in enumerate(loader, start=1):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item())
        num_steps += 1

        if max_batches is not None and batch_idx >= max_batches:
            break

    return running_loss / max(num_steps, 1)


def validate(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    patch_size: tuple[int, int, int],
    max_batches: int | None,
) -> float:
    """Run validation with sliding-window inference and return mean Dice."""
    model.eval()
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    post_pred, post_label = build_post_transforms()

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            # Validation volumes can be larger than the training patch. Sliding-window
            # inference makes the baseline less sensitive to memory limits and keeps the
            # evaluation setup aligned with the training crop size.
            logits = sliding_window_inference(
                images,
                roi_size=patch_size,
                sw_batch_size=1,
                predictor=model,
            )

            pred_list = [post_pred(x) for x in decollate_batch(logits)]
            label_list = [post_label(x) for x in decollate_batch(labels)]
            dice_metric(y_pred=pred_list, y=label_list)

            if max_batches is not None and batch_idx >= max_batches:
                break

    score = float(dice_metric.aggregate().item())
    dice_metric.reset()
    return score


def main() -> None:
    """Train and validate a minimal baseline to assess preprocessing quality."""
    args = parse_args()
    device = get_device()
    torch.manual_seed(args.seed)

    items = build_acdc_list()
    train_items, val_items = split_by_patient(items, n_splits=args.n_splits, fold=args.fold)

    train_transform = build_train_transform(
        target_spacing=tuple(args.target_spacing),
        patch_size=tuple(args.patch_size),
    )
    val_transform = build_val_transform(
        target_spacing=tuple(args.target_spacing),
        pad_size=tuple(args.patch_size),
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
    )

    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)

    history: list[dict[str, float | int]] = []
    best_val_dice = -1.0

    print(f"Device: {device}")
    print(f"Train cases: {len(train_items)} | Val cases: {len(val_items)}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            max_batches=args.max_train_batches,
        )

        epoch_metrics: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
        }

        if epoch % args.val_every == 0:
            val_dice = validate(
                model=model,
                loader=val_loader,
                device=device,
                patch_size=tuple(args.patch_size),
                max_batches=args.max_val_batches,
            )
            best_val_dice = max(best_val_dice, val_dice)
            epoch_metrics["val_dice"] = round(val_dice, 6)
            print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_dice={val_dice:.4f}")
        else:
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}")

        history.append(epoch_metrics)

    summary = {
        "config": {
            "fold": args.fold,
            "n_splits": args.n_splits,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
            "patch_size": list(args.patch_size),
            "target_spacing": list(args.target_spacing),
            "num_workers": args.num_workers,
            "cache_rate_train": args.cache_rate_train,
            "cache_rate_val": args.cache_rate_val,
            "val_every": args.val_every,
            "max_train_batches": args.max_train_batches,
            "max_val_batches": args.max_val_batches,
            "device": str(device),
        },
        "train_cases": len(train_items),
        "val_cases": len(val_items),
        "best_val_dice": None if best_val_dice < 0 else round(best_val_dice, 6),
        "history": history,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(f"Saved metrics to: {args.output}")


if __name__ == "__main__":
    main()
