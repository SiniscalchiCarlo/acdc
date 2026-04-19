from pathlib import Path

from config import (
    include_background_slices_2d,
    min_label_pixels_2d,
    patch_size_2d,
    preprocessing_mode,
    seed_2d,
    target_spacing_2d,
    test_path,
)

REPO_ROOT = Path(__file__).resolve().parent

# Model architecture for the checkpoint you want to test.
# Options: 'UNET' | 'ATTUNET' | 'SEGRESNET' | '25DATTUNET'
MODEL = "25DATTUNET"

# Explicit preprocessing mode used when preprocessing the raw test set on the fly.
# Options: '2d' | '2.5d'
PREPROCESSING_MODE = preprocessing_mode

# Path to the checkpoint file to evaluate.
MODEL_PATH = REPO_ROOT / "scripts" / "artifacts" / "models" / "baseline_model_2d5.pt"

# Root directory of the raw test dataset. Read from TEST_PATH in the environment.
TEST_ROOT = None if test_path is None else Path(test_path)

# Preprocessing settings for raw test volumes.
PATCH_SIZE = patch_size_2d
TARGET_SPACING = target_spacing_2d
INCLUDE_BACKGROUND_SLICES = include_background_slices_2d
MIN_LABEL_PIXELS = min_label_pixels_2d
SEED = seed_2d

# Evaluation dataloader settings.
BATCH_SIZE = 8
NUM_WORKERS = 0
CACHE_RATE_VAL = 0.0
PIN_MEMORY = False

# Optional debug limit. None evaluates the full test set.
MAX_VAL_BATCHES = None

# Output summary written after evaluation.
OUTPUT = REPO_ROOT / "artifacts" / "model_metrics" / "test_metrics.json"
