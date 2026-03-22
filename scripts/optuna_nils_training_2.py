import optuna
from optuna.pruners import MedianPruner
import sys
import torch
from pathlib import Path
from monai.losses import DiceCELoss
from monai.data import pad_list_data_collate

# Add repo root to path so imports work
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import functions from your training script
from nils_training_2 import (
    get_device,
    get_model,
    train_one_epoch,
    validate,
)
from src.load_data_2D import (
    build_preprocessed_2d_list,
    load_preprocessed_2d_manifest,
    split_by_patient,
    build_loaders,
)
from src.transforms_2D import (
    build_preprocessed_train_transform,
    build_preprocessed_val_transform,
)
import nils_training_2_config as cfg

# -----------------------------
# Warm start — path to best pretrained model checkpoint
# -----------------------------
PRETRAINED_MODEL_PATH = Path("artifacts") / "models" / "Baseline_ATTUNET.pt"

# Build data once outside objective (no need to reload every trial)
manifest = load_preprocessed_2d_manifest(cfg.PREPROCESSED_ROOT)
items = build_preprocessed_2d_list(cfg.PREPROCESSED_ROOT)
train_items, val_items = split_by_patient(items, n_splits=cfg.N_SPLITS, fold=cfg.FOLD)
train_transform = build_preprocessed_train_transform(patch_size=cfg.PATCH_SIZE)
val_transform = build_preprocessed_val_transform(patch_size=cfg.PATCH_SIZE)
collate_fn = pad_list_data_collate

def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 40])
    lambda_dice = trial.suggest_float("lambda_dice", 0.3, 1.0)
    lambda_ce = trial.suggest_float("lambda_ce", 0.3, 1.0)

    device = get_device()
    model = get_model(cfg).to(device)

    # Load pretrained weights as warm start if checkpoint exists
    if PRETRAINED_MODEL_PATH.exists():
        checkpoint = torch.load(PRETRAINED_MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = DiceCELoss(
        to_onehot_y=True,
        softmax=True,
        lambda_dice=lambda_dice,
        lambda_ce=lambda_ce,
    )

    train_loader, val_loader = build_loaders(
        train_items=train_items,
        val_items=val_items,
        train_transform=train_transform,
        val_transform=val_transform,
        batch_size=batch_size,
        num_workers=cfg.NUM_WORKERS,
        cache_rate_train=cfg.CACHE_RATE_TRAIN,
        cache_rate_val=cfg.CACHE_RATE_VAL,
        seed=cfg.SEED,
        pin_memory=cfg.PIN_MEMORY,
        collate_fn=collate_fn,
    )

    for epoch in range(10):
        train_one_epoch(model, train_loader, optimizer, loss_fn, device, max_batches=None)
        val_metrics = validate(model, val_loader, loss_fn, device, max_batches=None)
        trial.report(val_metrics["val_dice"], epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_metrics["val_dice"]


if __name__ == "__main__":
    storage = "sqlite:///optuna_study.db"
    study = optuna.create_study(
        direction="maximize",
        storage=storage,
        study_name="acdc_segmentation",
        load_if_exists=True,
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )

    # Enqueue known good values as the first trial so Optuna
    # starts from a strong baseline before exploring further
    study.enqueue_trial({
        "lr": 1e-3,             # current LR
        "weight_decay": 1e-5,   # current weight decay
        "batch_size": 40,       # current batch size
        "lambda_dice": 1.0,     # current lambda_dice
        "lambda_ce": 1.0,       # current lambda_ce
    })

    study.optimize(objective, n_trials=50)

    print("Best trial:")
    print(f"  val_dice: {study.best_trial.value}")
    print(f"  Params: {study.best_trial.params}")