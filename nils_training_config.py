from pathlib import Path

from config import (
    include_background_slices_2d,
    min_label_pixels_2d,
    patch_size_2d,
    preprocessed_2d_path,
    seed_2d,
    target_spacing_2d,
)

# Model-specific parameters (editable per-experiment)
MODEL_IN_CHANNELS = 1
MODEL_OUT_CHANNELS = 4
# network feature channels for UNet encoder/decoder
MODEL_CHANNELS = (16, 32, 64, 128, 256)
# strides between feature levels (length should be len(MODEL_CHANNELS)-1)
MODEL_STRIDES = (2, 2, 2, 2)
MODEL_NUM_RES_UNITS = 2

# DiceLoss parameters
DICE_TO_ONEHOT_Y = True
DICE_SOFTMAX = True
DICE_INCLUDE_BACKGROUND = True
DICE_REDUCTION = "mean"

# Optimizer parameters
LR = 1e-3
WEIGHT_DECAY = 1e-5

# Scheduler
SCHEDULER_PATIENCE = 6
SCHEDULER_FACTOR = 0.5
SCHEDULER_MIN_LR = 1e-6

# Amount of epochs to train for (sanity test)
EPOCHS = 1

# Data loader / batching
BATCH_SIZE = 32
NUM_WORKERS = 2
CACHE_RATE_TRAIN = 1.0
CACHE_RATE_VAL = 1.0
PIN_MEMORY = False

# Use the project-level seed by default
SEED = seed_2d

# Patch size (overrides cfg.PATCH_SIZE for model runs)
PATCH_SIZE = patch_size_2d

# Validation / training schedule
VAL_EVERY = 1
EARLY_STOP_PATIENCE = 20

# Path to save model for this experiment (falls back to cfg.MODEL_OUTPUT if not set)
MODEL_OUTPUT: Path | None = Path(r"C:\Users\Nils\Documents\Studie Nils\Deep Learning 3D Images\ACDC_data\artifacts\models")