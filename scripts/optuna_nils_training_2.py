import optuna
from optuna.pruners import MedianPruner
import sys
import torch
from pathlib import Path
from monai.losses import DiceCELoss
from monai.data import pad_list_data_collate

# -----------------------------
# Add repo root to path
# -----------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
# Configurable stopping criteria
# -----------------------------
MAX_TRIALS = 50
TIMEOUT = 60 * 60 * 4  # 4 hours
EARLY_STOP_PATIENCE = 10  # stop if no improvement in N trials

# -----------------------------
# Warm start
# -----------------------------
PRETRAINED_MODEL_PATH = Path("artifacts") / "models" / "DFL_ATTUNET.pt"

# -----------------------------
# Build data once
# -----------------------------
manifest = load_preprocessed_2d_manifest(cfg.PREPROCESSED_ROOT)
items = build_preprocessed_2d_list(cfg.PREPROCESSED_ROOT)
train_items, val_items = split_by_patient(items, n_splits=cfg.N_SPLITS, fold=cfg.FOLD)

train_transform = build_preprocessed_train_transform(patch_size=cfg.PATCH_SIZE)
val_transform = build_preprocessed_val_transform(patch_size=cfg.PATCH_SIZE)

collate_fn = pad_list_data_collate

# -----------------------------
# Objective
# -----------------------------
def objective(trial):
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    lambda_dice = trial.suggest_float("lambda_dice", 0.3, 2.0)
    lambda_ce = trial.suggest_float("lambda_ce", 0.3, 2.0)

    device = get_device()
    model = get_model(cfg).to(device)

    if PRETRAINED_MODEL_PATH.exists():
        checkpoint = torch.load(PRETRAINED_MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR, weight_decay=weight_decay)

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
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        cache_rate_train=cfg.CACHE_RATE_TRAIN,
        cache_rate_val=cfg.CACHE_RATE_VAL,
        seed=cfg.SEED,
        pin_memory=cfg.PIN_MEMORY,
        collate_fn=collate_fn,
    )

    for epoch in range(30):
        train_one_epoch(model, train_loader, optimizer, loss_fn, device, max_batches=None)
        val_metrics = validate(model, val_loader, loss_fn, device, max_batches=None)

        trial.report(val_metrics["val_dice"], epoch)
        trial.set_user_attr("val_hd95", val_metrics["val_hd95"])

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_metrics["val_dice"]

# -----------------------------
# Early stopping callback
# -----------------------------
class EarlyStoppingCallback:
    def __init__(self, patience):
        self.patience = patience
        self.best_value = None
        self.no_improve_count = 0

    def __call__(self, study, trial):
        if study.best_value is None:
            return

        if self.best_value is None or study.best_value > self.best_value:
            self.best_value = study.best_value
            self.no_improve_count = 0
        else:
            self.no_improve_count += 1

        if self.no_improve_count >= self.patience:
            print(f"\nStopping early: no improvement in {self.patience} trials.")
            study.stop()

# -----------------------------
# Run study
# -----------------------------
if __name__ == "__main__":
    storage = "sqlite:///optuna_study.db"

    study = optuna.create_study(
        direction="maximize",
        storage=storage,
        study_name="acdc_segmentation",
        load_if_exists=True,
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )

    study.enqueue_trial({
        "weight_decay": 1e-4,
        "lambda_dice": 1.0,
        "lambda_ce": 1.0,
    })

    study.optimize(
        objective,
        n_trials=MAX_TRIALS,
        timeout=TIMEOUT,
        callbacks=[EarlyStoppingCallback(EARLY_STOP_PATIENCE)],
    )

    print("Best trial:")
    print(f"  val_dice: {study.best_trial.value}")
    print(f"  val_hd95: {study.best_trial.user_attrs.get('val_hd95', 'N/A')}")
    print(f"  Params: {study.best_trial.params}")