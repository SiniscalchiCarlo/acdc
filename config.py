from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from src.mode_compatibility import normalize_preprocessing_mode


load_dotenv()


# -----------------------------
# Important Parameters
# -----------------------------
# Options: "2d" | "2.5d"
PIPELINE_MODE = normalize_preprocessing_mode("2.5d")

MODE_TAG_BY_PIPELINE = {
    "2d": "2d",
    "2.5d": "2d5",
}

MODE_TAG = MODE_TAG_BY_PIPELINE[PIPELINE_MODE]

# Main architecture used by train/test by default.
# Options: 'UNET' | 'ATTUNET' | 'SEGRESNET' | '25DATTUNET'
MODEL = '25DATTUNET' 

# Raw dataset locations. You need to setup them in .env
DATASET_PATH = os.getenv("DATASET_PATH")
TEST_PATH = os.getenv("TEST_PATH")

# Main generated dataset path derived from the active mode.
PREPROCESSED_DATA_PATH = f"artifacts/preprocessed_{MODE_TAG}"
PREPROCESSED_ROOT = Path(PREPROCESSED_DATA_PATH)

# Core preprocessing parameters shared across preprocess, train, and test.
PATCH_SIZE = (320, 320)
TARGET_SPACING = (1.25, 1.25, -1.0)
INCLUDE_BACKGROUND_SLICES = True
MIN_LABEL_PIXELS = 1
SEED = 42

# Main train/test outputs derived from the active mode.
MODEL_OUTPUT_DIR = "artifacts/models"
MODEL_OUTPUT = Path(MODEL_OUTPUT_DIR) / f"baseline_model_{MODE_TAG}.pt"
OUTPUT = Path("artifacts") / "model_metrics" / f"baseline_metrics_{MODE_TAG}.json"
TEST_OUTPUT = Path("artifacts") / "model_metrics" / f"test_metrics_{MODE_TAG}.json"
MODEL_PATH = MODEL_OUTPUT
PREPROCESSING_MODE = PIPELINE_MODE
TEST_ROOT = None if TEST_PATH is None else Path(TEST_PATH)


# -----------------------------
# Training Parameters
# -----------------------------
FOLD = 0
N_SPLITS = 5
VAL_SIZE = 0.2

EPOCHS = 100
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 2.8e-5

NUM_WORKERS = 10
PIN_MEMORY = True
CACHE_RATE_TRAIN = 0.0
CACHE_RATE_VAL = 0.0

VAL_EVERY = 1
EARLY_STOP_PATIENCE = 25
SCHEDULER_PATIENCE = 8
SCHEDULER_FACTOR = 0.5
SCHEDULER_MIN_LR = 1e-6

LAMBDA_DICE = 1.5
LAMBDA_CE = 0.9
LOSS_FUNCTION = "DiceCE"
DYNAMIC_CLASS_WEIGHTS = True

MAX_TRAIN_BATCHES = None
MAX_VAL_BATCHES = None


# -----------------------------
# Testing Parameters
# -----------------------------
TEST_BATCH_SIZE = 8
TEST_NUM_WORKERS = 0
TEST_PIN_MEMORY = False
TEST_CACHE_RATE_VAL = 0.0
TEST_MAX_VAL_BATCHES = None


# -----------------------------
# Logging Parameters
# -----------------------------
WANDB_ENABLED = True
WANDB_PROJECT = "acdc-segmentation"
WANDB_ENTITY = "n-x-stuurop-university-of-twente"
WANDB_RUN_NAME = None
WANDB_MODE = "online"


# -----------------------------
# QC Parameters
# -----------------------------
PREPROCESS_QC_LIMIT_2D = 20
PREPROCESS_QC_OUTPUT_JSON_2D = "artifacts/preprocess_qc_report_2d.json"
PREPROCESS_QC_OUTPUT_FIGURES_2D = "artifacts/preprocess_visual_qc_2d"
PREPROCESS_QC_OUTPUT_JSON_2D5 = "artifacts/preprocess_qc_report_2d5.json"
PREPROCESS_QC_OUTPUT_FIGURES_2D5 = "artifacts/preprocess_visual_qc_2d5"
