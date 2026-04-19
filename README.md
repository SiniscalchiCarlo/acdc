# ACDC Segmentation

This repository contains a MONAI-based cardiac MRI segmentation workflow for the ACDC dataset. The current pipeline is organized around four steps:

1. preprocess the raw dataset into offline slice files
2. run a lightweight preprocessing Quality and Control script (QC)
3. train a model on the preprocessed dataset
4. test a saved checkpoint on a raw test dataset

The preprocessing mode is explicit through `PREPROCESSING_MODE`:

- `2d`: one slice per sample
- `2.5d`: previous, center, next slice stacked as 3 channels

The preprocessing mode is set in [config.py](/home/carlo/UT/deep_med/acdc/config.py), not in `.env`. The default is `2.5d`, so the default training setup in [training_config.py](/home/carlo/UT/deep_med/acdc/training_config.py) uses `MODEL = "25DATTUNET"`.

**Repository Layout**
- [config.py](/home/carlo/UT/deep_med/acdc/config.py): global paths, preprocessing defaults, and QC output locations
- [training_config.py](/home/carlo/UT/deep_med/acdc/training_config.py): training hyperparameters, model choice, and training output paths
- [test_config.py](/home/carlo/UT/deep_med/acdc/test_config.py): checkpoint path, test dataset path handling, and test output path
- [scripts/preprocess_dataset.py](/home/carlo/UT/deep_med/acdc/scripts/preprocess_dataset.py): offline preprocessing
- [scripts/preprocess_qc.py](/home/carlo/UT/deep_med/acdc/scripts/preprocess_qc.py): preprocessing QC with figures and JSON summary
- [scripts/training.py](/home/carlo/UT/deep_med/acdc/scripts/training.py): training entrypoint
- [scripts/test_model.py](/home/carlo/UT/deep_med/acdc/scripts/test_model.py): checkpoint evaluation entrypoint

**Setup**
Run everything from the repository root.

Install dependencies with `uv`:

```bash
uv sync
```

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Or run commands directly with `uv run python ...` if you prefer not to activate `.venv`.

**Environment Variables**
Create a local `.env` file in the repository root. The project loads it automatically through `python-dotenv`.

Use `.env.example` as the starting point:

```dotenv
# Required: raw ACDC training dataset root
DATASET_PATH=/path/to/acdc/database/training

# Required for testing: raw ACDC test dataset root
TEST_PATH=/path/to/acdc/database/testing

# where offline-preprocessed slices and manifest are written
PREPROCESSED_DATA_PATH=artifacts/preprocessed_data

# QC report outputs
PREPROCESS_QC_OUTPUT_JSON_2D=artifacts/preprocess_qc_report_2d.json
PREPROCESS_QC_OUTPUT_FIGURES_2D=artifacts/preprocess_visual_qc_2d
```

What each variable does:

- `DATASET_PATH`: raw training set used by preprocessing and QC
- `TEST_PATH`: raw test set used by `scripts/test_model.py`
- `PREPROCESSED_DATA_PATH`: output folder for `.npz` slices plus `manifest.json`
- `PREPROCESS_QC_OUTPUT_JSON_2D`: JSON summary written by the QC step
- `PREPROCESS_QC_OUTPUT_FIGURES_2D`: folder with QC figures

**Config Files**
There are three places to check before running the pipeline:

- [config.py](/home/carlo/UT/deep_med/acdc/config.py): shared defaults for spacing, patch size, random seed, and environment-driven paths
- `preprocessing_mode` in [config.py](/home/carlo/UT/deep_med/acdc/config.py): set this to `2d` or `2.5d`
- [training_config.py](/home/carlo/UT/deep_med/acdc/training_config.py): training-specific settings such as `MODEL`, `EPOCHS`, `BATCH_SIZE`, `WANDB_ENABLED`, `OUTPUT`, and `MODEL_OUTPUT`
- [test_config.py](/home/carlo/UT/deep_med/acdc/test_config.py): test-specific settings such as `MODEL`, `PREPROCESSING_MODE`, `MODEL_PATH`, and `OUTPUT`

Important notes:

- Keep `training_config.py` and the preprocessed dataset aligned. The training script checks that manifest settings match the current config.
- `25DATTUNET` requires `PREPROCESSING_MODE=2.5d`
- `UNET`, `ATTUNET`, and `SEGRESNET` require `PREPROCESSING_MODE=2d`
- Before testing, update `MODEL_PATH` in [test_config.py](/home/carlo/UT/deep_med/acdc/test_config.py) so it points to the checkpoint you want to evaluate.

**Execution Order**
Use the following order (inside active venv).

1. Preprocess the dataset

```bash
python scripts/preprocess_dataset.py
```

This reads the raw training dataset from `DATASET_PATH` and writes:

- preprocessed samples to `PREPROCESSED_DATA_PATH/samples`
- a manifest to `PREPROCESSED_DATA_PATH/manifest.json`

2. Run preprocessing QC

```bash
python scripts/preprocess_qc.py
```

This samples slices from the raw training set, applies the same preprocessing and augmentation logic, and writes:

- a JSON report to `PREPROCESS_QC_OUTPUT_JSON_2D`
- figures to `PREPROCESS_QC_OUTPUT_FIGURES_2D`

3. Train

Before training, review [training_config.py](/home/carlo/UT/deep_med/acdc/training_config.py), especially:

- `MODEL`
- `EPOCHS`
- `BATCH_SIZE`
- `WANDB_ENABLED`
- `MODEL_OUTPUT`

Then run:

```bash
python scripts/training.py
```

By default this writes:

- metrics to `artifacts/model_metrics/baseline_metrics_2d5.json`
- checkpoint to `artifacts/models/baseline_model_2d5.pt`

4. Test

Before testing:

- make sure `TEST_PATH` is set in `.env`
- set `MODEL` in [test_config.py](/home/carlo/UT/deep_med/acdc/test_config.py) to match the trained checkpoint
- set `PREPROCESSING_MODE` in [test_config.py](/home/carlo/UT/deep_med/acdc/test_config.py) so it matches the checkpoint architecture
- set `MODEL_PATH` in [test_config.py](/home/carlo/UT/deep_med/acdc/test_config.py) to the checkpoint you want to evaluate

Then run:

```bash
python scripts/test_model.py
```

By default this writes:

- test metrics to `artifacts/model_metrics/test_metrics.json`

**Minimal End-to-End Command List**
```bash
uv sync
source .venv/bin/activate
python scripts/preprocess_dataset.py
python scripts/preprocess_qc.py
python scripts/training.py
python scripts/test_model.py
```
