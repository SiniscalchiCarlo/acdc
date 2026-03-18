# Preprocessing Justification

## Goal
This file records why preprocessing choices are made for the ACDC cardiac MRI segmentation project. It is meant to prevent parameter choices from becoming arbitrary and to tie every important setting to either dataset properties or QC evidence.

## Current Decisions

### Semi-isotropic resampling
We resample only the in-plane spacing to `1.25 x 1.25` and keep native `z` spacing with `-1.0`.

Why:
- ACDC short-axis stacks are anisotropic. The in-plane spacing is around `1.37-1.79 mm` in the sampled QC cases, while the slice spacing is `5-10 mm`.
- Forcing isotropic spacing in `z` would create artificial slices by interpolation and would blur or distort anatomy across slices.
- Resampling `x/y` only makes the in-plane anatomy more consistent while preserving the true through-plane sampling.

Evidence from the 20-case QC run:
- Raw spacing summary: `min [1.3672, 1.3672, 5.0]`, `median [1.4648, 1.4648, 10.0]`, `max [1.7857, 1.7857, 10.0]`
- Processed spacing summary: `min [1.25, 1.25, 5.0]`, `median [1.25, 1.25, 10.0]`, `max [1.25, 1.25, 10.0]`

### Interpolation modes
We use `bilinear` interpolation for images and `nearest` interpolation for labels.

Why:
- MRI intensities are continuous, so smooth interpolation is appropriate for the image.
- Segmentation labels are categorical. Any non-nearest interpolation would create invalid class values at boundaries.

Evidence from QC:
- Observed processed label sets stayed `[[0, 1, 2, 3]]` across the sampled cases.
- No warning cases were reported for label corruption.

### Orientation normalization
We keep `Orientationd(..., axcodes="RAS")`.

Why:
- Spatial transforms are easier to reason about when all volumes share a common orientation.
- This reduces the chance of silent inconsistencies across scanners or export pipelines.

### Intensity normalization
We use per-volume z-score normalization on nonzero voxels only.

Why:
- MRI intensities do not have a fixed absolute scale across acquisitions.
- Restricting the statistics to nonzero voxels avoids background dominating the mean and standard deviation.

### Visual quality control
We now keep a visualization script to inspect raw and preprocessed overlays before trusting training metrics.

Why:
- Numeric summaries can show that shapes and labels are preserved, but they do not reveal whether anatomy looks distorted after orientation or resampling.
- Overlay figures make it easier to catch misalignment between image and label or visually implausible interpolation artifacts.

Current workflow:
- Use `scripts/preprocess_qc.py` for numeric checks.
- Use `scripts/preprocess_visual_qc.py` for slice overlays saved to `artifacts/preprocess_visual_qc/`.
- Use the same visualization script with augmentation enabled to inspect whether random training perturbations remain anatomically plausible.

Why augmentation needs its own visualization:
- Deterministic preprocessing should be judged on anatomical fidelity and label alignment.
- Training augmentation should be judged on realism and robustness, because transforms such as flips, affine perturbations, contrast changes, noise, and smoothing are not expected to appear in the deterministic pipeline.

## Issues Identified

### Train/validation preprocessing mismatch
Training and validation originally used different `z` spacing policies. This was a correctness issue because validation would no longer reflect the training data distribution. The code was updated so both paths now default to the same semi-isotropic spacing policy.

### No foreground crop in the active pipeline
The active 2D pipeline does not apply image-based foreground cropping.

Why:
- The available crop heuristic was not tightly localizing the heart; it mostly removed empty margins around the torso.
- Keeping a single no-crop path avoids train/preprocess drift and makes offline preprocessing manifests easier to validate.
- Padding to a fixed `patch_size_2d` is enough to keep model inputs consistent for the current baseline.

## What To Measure Next
- Label bounding-box size after preprocessing across the dataset
- Whether the current training patch size `192 x 192 x 16` covers the heart plus margin in nearly all cases
- Validation metrics from controlled ablations:
  - baseline loading only
  - `+ normalization`
  - `+ normalization + resampling`
  - `+ normalization + resampling + pad`

## Baseline Training Plan
We now keep a minimal training baseline to test whether the current preprocessing is good enough in practice.

Why a baseline is needed before more preprocessing changes:
- Geometry checks can show that the pipeline is consistent, but they cannot show whether the model learns useful segmentations.
- A safe but loose preprocessing pipeline is still a valid baseline if it trains stably and gives sensible validation Dice.
- Future preprocessing or augmentation changes should be compared against one fixed reference setup rather than against intuition.

Why the baseline is intentionally simple:
- One architecture: a modest 3D UNet
- One split policy: patient-wise GroupKFold
- One validation method: Dice on a held-out fold
- Sliding-window inference during validation so evaluation is compatible with full preprocessed volumes

Why some engineering defaults are conservative:
- `num_workers=0` reduces debugging noise and makes the first baseline less dependent on local multiprocessing behavior.
- `cache_rate=0.0` reduces memory-related confounders while we are still validating the data pipeline.
- A small number of epochs or capped batches can be used first as a smoke baseline before spending time on a longer run.

## Change Policy
Whenever preprocessing parameters change, this file should be updated with:
- what changed
- why it changed
- what evidence motivated the change
- what effect was observed in QC or validation
