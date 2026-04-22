# ACDC Segmentation

This repository contains a MONAI-based cardiac MRI segmentation workflow for the ACDC dataset. It supports two slice pipelines:

- `2d`: one slice per sample
- `2.5d`: previous, center, and next slice stacked as 3 channels

The pipeline has four steps:

1. preprocess the raw training dataset into offline slice files
2. run preprocessing QC
3. train a model on the preprocessed dataset
4. test a saved checkpoint on the raw test dataset

**Configuration**

There is one user-facing config file: config.py

The top of `config.py` contains the most important parameters to run the scripts:

- `PIPELINE_MODE`
- `MODEL`
- `DATASET_PATH`
- `TEST_PATH`
- `PREPROCESSED_ROOT`
- `MODEL_OUTPUT`

The rest of the file contains the training, testing, logging, and QC settings.

`config.py` is the source of truth for run parameters. `.env` is only used for external dataset locations.

**Mode Selection**

Set:

```python
pipeline_mode = "2d"
```

or:

```python
pipeline_mode = "2.5d"
```

Everything else follows that default:

- `PREPROCESSING_MODE`
- default training model
- default preprocessed dataset path
- default checkpoint filename
- default metrics filename

Current default mapping:

| `pipeline_mode` | default model | input format | default preprocessed path | default checkpoint |
| --- | --- | --- | --- | --- |
| `2d` | `UNET` | single slice | `artifacts/preprocessed_2d` | `artifacts/models/baseline_model_2d.pt` |
| `2.5d` | `25DATTUNET` | 3 stacked slices | `artifacts/preprocessed_2d5` | `artifacts/models/baseline_model_2d5.pt` |

**Model Compatibility**

- `UNET`, `ATTUNET`, and `SEGRESNET` require `2d`
- `25DATTUNET` requires `2.5d`

The compatibility check is enforced in [src/mode_compatibility.py](/home/carlo/Download/acdc/src/mode_compatibility.py:1) and validated again at training time against the saved preprocess manifest.

**Repository Layout**

- [config.py](/home/carlo/Download/acdc/config.py:1): single user-facing config for mode, data, train, test, outputs, logging, and QC
- [scripts/preprocess_dataset.py](/home/carlo/Download/acdc/scripts/preprocess_dataset.py:1): offline preprocessing
- [scripts/preprocess_qc.py](/home/carlo/Download/acdc/scripts/preprocess_qc.py:1): preprocessing QC
- [scripts/training.py](/home/carlo/Download/acdc/scripts/training.py:1): training entrypoint
- [scripts/optuna_training.py](/home/carlo/Download/acdc/scripts/optuna_training.py:1): Optuna hyperparameter search entrypoint
- [scripts/test_model.py](/home/carlo/Download/acdc/scripts/test_model.py:1): checkpoint evaluation entrypoint


**Setup**

Run everything from the repository root.

Install dependencies:

```bash
uv sync
```

Activate the virtual environment:

```bash
source .venv/bin/activate
```

**Environment Variables**

Create a local `.env` file in the repository root. The project loads it automatically through `python-dotenv`.

Example:

```dotenv
DATASET_PATH=/path/to/acdc/database/training
TEST_PATH=/path/to/acdc/database/testing
```

What each variable does:

- `DATASET_PATH`: raw training set used by preprocessing and QC
- `TEST_PATH`: raw test set used by `scripts/test_model.py`

All generated paths such as preprocessed data, checkpoints, metrics, and QC outputs are defined in [config.py](/home/carlo/Download/acdc/config.py:1), not in `.env`.


**Example**
So if you want to run two experiments, both using UNET, one 2D and the otherone 2.5D you have to:

1. For UNET on 2d

- In config.py:16, set PIPELINE_MODE = "2d"
- In config.py:29, set MODEL = "UNET"
- Run:

```bash
python scripts/preprocess_dataset.py
python scripts/preprocess_qc.py
python scripts/training.py
python scripts/test_model.py
```

2. For 2.5d

- In config.py:16, set PIPELINE_MODE = "2.5d"
- In config.py:29, set MODEL = "25DATTUNET"
- Run again:

```bash
python scripts/preprocess_dataset.py
python scripts/preprocess_qc.py
python scripts/training.py
python scripts/test_model.py
```

**Optuna Hyperparameter Search**

To run the Optuna search script, first preprocess the dataset for the active mode and then run:

```bash
python scripts/optuna_training.py
```

Optuna tunes the learning rate, weight decay, mini-batch size, and the relative weighting of the Dice and cross-entropy terms in the combined loss.
