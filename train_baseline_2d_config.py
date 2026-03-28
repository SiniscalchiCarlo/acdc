from pathlib import Path

from config import (
    include_background_slices_2d,
    min_label_pixels_2d,
    patch_size_2d,
    preprocessed_2d_path,
    seed_2d,
    target_spacing_2d,
)

VAL_SIZE = 0.2
EPOCHS = 100
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-5
SEED = seed_2d

PATCH_SIZE = patch_size_2d
TARGET_SPACING = target_spacing_2d
INCLUDE_BACKGROUND_SLICES = include_background_slices_2d
MIN_LABEL_PIXELS = min_label_pixels_2d

NUM_WORKERS = 2
CACHE_RATE_TRAIN = 1.0
CACHE_RATE_VAL = 1.0
PIN_MEMORY = False

VAL_EVERY = 1
EARLY_STOP_PATIENCE = 20
SCHEDULER_PATIENCE = 6
SCHEDULER_FACTOR = 0.5
SCHEDULER_MIN_LR = 1e-6

MAX_TRAIN_BATCHES = None
MAX_VAL_BATCHES = None

PREPROCESSED_ROOT = Path(preprocessed_2d_path)
OUTPUT = Path("artifacts") / "baseline_metrics_2d.json"
MODEL_OUTPUT = Path("artifacts") / "baseline_model_2d.pt"
