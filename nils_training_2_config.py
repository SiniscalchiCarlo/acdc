from pathlib import Path

from config import (
    include_background_slices_2d,
    min_label_pixels_2d,
    patch_size_2d,
    preprocessed_2d_path,
    seed_2d,
    target_spacing_2d,
)

# -----------------------------
# Reproducibility / splitting
# -----------------------------
# Random seed for reproducible runs (affects NumPy and PyTorch RNGs)
SEED = seed_2d

# Which CV fold to run (0-based index). Change to run a different split.
FOLD = 0 # No folding, just one split for training and validation. Set to 0 for simplicity.

# Number of patient-level splits for cross-validation (used by `split_by_patient`).
N_SPLITS = 5

# -----------------------------
# Model / architecture
# -----------------------------
# Model choice string used by `get_model()` to construct the network.
# Options: 'UNET' | 'ATTUNET' | 'SEGRESNET'
MODEL = 'ATTUNET'

# -----------------------------
# Data / preprocessing
# -----------------------------
# Path to offline preprocessed dataset. If `None`, raw data will be preprocessed on-the-fly.
PREPROCESSED_ROOT = Path(preprocessed_2d_path)

# Patch / crop size used by spatial transforms: (height, width).
PATCH_SIZE = patch_size_2d

# Desired voxel spacing used when resampling during preprocessing.
TARGET_SPACING = target_spacing_2d

# Whether to include slices that contain only background (no labels).
INCLUDE_BACKGROUND_SLICES = include_background_slices_2d

# Minimum number of foreground pixels required for a slice to be kept.
MIN_LABEL_PIXELS = min_label_pixels_2d

# -----------------------------
# Training hyperparameters
# -----------------------------
# Total number of training epochs to run.
EPOCHS = 1

# Number of samples per training batch.
BATCH_SIZE = 32

# Initial learning rate for the optimizer (Adam).
LR = 1e-3

# Weight decay (L2 regularization) applied by the optimizer.
WEIGHT_DECAY = 1e-5

# -----------------------------
# Dataloader / performance
# -----------------------------
# Number of subprocesses for DataLoader workers.
NUM_WORKERS = 2

# If True, DataLoader will use pinned memory to accelerate CUDA transfers.
PIN_MEMORY = True

# Fraction of training dataset to cache in memory (0.0 to 1.0). 1.0 caches all.
CACHE_RATE_TRAIN = 1.0

# Fraction of validation dataset to cache in memory.
CACHE_RATE_VAL = 1.0

# -----------------------------
# Scheduler / early stopping
# -----------------------------
# Run validation every N epochs.
VAL_EVERY = 1

# Number of consecutive validations without improvement before early stopping.
EARLY_STOP_PATIENCE = 20

# Number of validations without improvement before LR scheduler reduces LR.
SCHEDULER_PATIENCE = 6

# Multiplicative factor for LR reduction (new_lr = lr * SCHEDULER_FACTOR).
SCHEDULER_FACTOR = 0.5

# Minimum LR value the scheduler will reduce to.
SCHEDULER_MIN_LR = 1e-6

# -----------------------------
# Debug / limits
# -----------------------------
# Limit number of training batches per epoch (useful for quick debugging). None = no limit.
MAX_TRAIN_BATCHES = None

# Limit number of validation batches per run. None = no limit.
MAX_VAL_BATCHES = None

# -----------------------------
# Outputs
# -----------------------------
# JSON file where run summary and metrics are written.

OUTPUT = Path("artifacts") / "model_metrics" / "baseline_metrics_2d.json"

# File path for saving the final model checkpoint (.pt).
MODEL_OUTPUT = Path("artifacts") / "models" / "baseline_model_2d.pt"

# -----------------------------
# Weights & Biases logging
# -----------------------------
# Enable/disable W&B experiment tracking.
WANDB_ENABLED = True

# W&B project and optional entity/team.
WANDB_PROJECT = "acdc-segmentation"
WANDB_ENTITY = None

# W&B run name. Set to None to auto-generate.
WANDB_RUN_NAME = None

# W&B mode: "online", "offline", or "disabled".
WANDB_MODE = "online"