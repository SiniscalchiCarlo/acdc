import optuna
from optuna.pruners import MedianPruner
import sys
import torch
from pathlib import Path
from monai.data import pad_list_data_collate
from tqdm import tqdm

# -----------------------------
# Add repo root to path
# -----------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training import (
    get_device,
    get_model,
    train_one_epoch,
    validate,
    compute_class_weights,
    build_loss_fn,
)
from src.load_data_2D import (
    build_loaders,
    build_preprocessed_dataset_list,
    load_preprocessed_manifest,
    split_by_patient,
)
from src.transforms_2D import (
    build_preprocessed_train_transform,
    build_preprocessed_val_transform,
)
import training_config as cfg

# -----------------------------
# Configurable stopping criteria
# -----------------------------
MAX_TRIALS = 75
TIMEOUT = 60 * 60 * 4  # 4 hours
EARLY_STOP_PATIENCE = 10  # stop if no improvement in N trials

# -----------------------------
# Warm start
# Note: Use a model checkpoint that matches the architecture (e.g., 25DATTUNET for 2.5D or standard for 2D)
# Set to None to start from scratch
PRETRAINED_MODEL_PATH = Path("artifacts") / "models" / "ATTENUNET_2.5D.pt"


# -----------------------------
# Objective
# -----------------------------
def objective(trial):
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    lambda_dice = trial.suggest_float("lambda_dice", 0.3, 2.0)
    lambda_ce = trial.suggest_float("lambda_ce", 0.3, 2.0)

    device = get_device()
    model = get_model(cfg).to(device)

    if PRETRAINED_MODEL_PATH is not None and PRETRAINED_MODEL_PATH.exists():
        checkpoint = torch.load(PRETRAINED_MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    # Set seed for reproducibility within each trial
    torch.manual_seed(cfg.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.SEED)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=cfg.SCHEDULER_FACTOR,
        patience=cfg.SCHEDULER_PATIENCE,
        min_lr=cfg.SCHEDULER_MIN_LR,
    )

    # Start with equal weights for all 4 classes
    class_weights = torch.ones(4, dtype=torch.float32).to(device)
    loss_fn = build_loss_fn(class_weights)
    # Override loss function lambdas with trial suggestions
    if hasattr(loss_fn, 'lambda_dice'):
        loss_fn.lambda_dice = lambda_dice
    if hasattr(loss_fn, 'lambda_ce'):
        loss_fn.lambda_ce = lambda_ce
    if hasattr(loss_fn, 'lambda_focal'):
        loss_fn.lambda_focal = lambda_ce

    best_hd95 = float("inf")

    # Train for cfg.EPOCHS with progress bar per trial
    for epoch in tqdm(range(cfg.EPOCHS), desc=f"Trial {trial.number+1}/{MAX_TRIALS}", unit="epoch", leave=False):
        train_one_epoch(model, train_loader, optimizer, loss_fn, device, max_batches=None)
        val_metrics = validate(model, val_loader, loss_fn, device, max_batches=None)

        trial.report(val_metrics["val_dice"], epoch)

        # Track the best HD95 seen so far (lower is better)
        if val_metrics["val_hd95"] < best_hd95:
            best_hd95 = val_metrics["val_hd95"]
            trial.set_user_attr("best_val_hd95", best_hd95)

        # Update scheduler
        scheduler.step(val_metrics["val_dice"])

        # Dynamically update class weights if enabled
        if cfg.DYNAMIC_CLASS_WEIGHTS:
            class_weights = compute_class_weights(val_metrics["val_dice_per_class"], device)
            loss_fn = build_loss_fn(class_weights)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_metrics["val_dice"]


# -----------------------------
# Early stopping callback
# -----------------------------
class EarlyStoppingCallback:
    """Stop study if no improvement in completed trials (ignore pruned trials)."""
    def __init__(self, patience):
        self.patience = patience
        self.best_value = None
        self.no_improve_count = 0

    def __call__(self, study, trial):
        # Only count completed trials for early stopping (ignore pruned ones)
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return

        if study.best_value is None:
            return

        if self.best_value is None or study.best_value > self.best_value:
            self.best_value = study.best_value
            self.no_improve_count = 0
        else:
            self.no_improve_count += 1

        if self.no_improve_count >= self.patience:
            print(f"\nStopping early: no improvement in {self.patience} completed trials.")
            study.stop()


# -----------------------------
# Entry point — all data/study init is guarded here
# so that Windows multiprocessing workers do NOT re-run
# this code when they re-import the module on spawn.
# -----------------------------
if __name__ == "__main__":

    # Build data once
    manifest = load_preprocessed_manifest(cfg.PREPROCESSED_ROOT)
    items = build_preprocessed_dataset_list(cfg.PREPROCESSED_ROOT)
    train_items, val_items = split_by_patient(items, val_size=cfg.VAL_SIZE, seed=cfg.SEED)

    train_transform = build_preprocessed_train_transform(patch_size=cfg.PATCH_SIZE)
    val_transform = build_preprocessed_val_transform(patch_size=cfg.PATCH_SIZE)

    collate_fn = pad_list_data_collate

    # Build data loaders once (reused across all trials)
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

    # Run study
    storage = "sqlite:///optuna_study.db"

    study = optuna.create_study(
        direction="maximize",
        storage=storage,
        study_name="acdc_segmentation_2.5D",
        load_if_exists=True,
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )

    # Enqueue initial trial with config-aligned defaults
    #study.enqueue_trial({
   #     "weight_decay": cfg.WEIGHT_DECAY,
   #     "learning_rate": cfg.LR,
    #    "lambda_dice": cfg.LAMBDA_DICE,
   #     "lambda_ce": cfg.LAMBDA_CE,
    #})

    trial_bar = tqdm(total=MAX_TRIALS, desc="Optuna Trials", unit="trial", position=0)

    def optuna_callback(study, trial):
        trial_bar.update(1)
        EarlyStoppingCallback(EARLY_STOP_PATIENCE)(study, trial)

    study.optimize(
        objective,
        n_trials=MAX_TRIALS,
        timeout=TIMEOUT,
        callbacks=[optuna_callback],
    )

    trial_bar.close()

    print("\n" + "=" * 60)
    print(f"Best trial (Trial {study.best_trial.number}):")
    print(f"  val_dice: {study.best_trial.value:.6f}")
    print(f"  best_val_hd95: {study.best_trial.user_attrs.get('best_val_hd95', 'N/A')}")
    print(f"  Params:")
    for key, val in study.best_trial.params.items():
        print(f"    {key}: {val}")
    print("=" * 60)
