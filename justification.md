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

## Issues Identified

### Train/validation preprocessing mismatch
Training and validation originally used different `z` spacing policies. This was a correctness issue because validation would no longer reflect the training data distribution. The code was updated so both paths now default to the same semi-isotropic spacing policy.

### Cropping is still loose
Current QC suggests that image-based foreground cropping is removing obvious empty space, but not tightly localizing the heart.

Evidence from the 20-case QC run:
- Processed shape summary: `min [206, 213, 16]`, `median [253.5, 279, 16]`, `max [319, 319, 16]`
- Processed label bbox summary: `min [55, 55, 6]`, `median [84, 81.5, 9]`, `max [122, 100, 16]`
- Processed foreground fraction: `min 0.007761`, `median 0.025005`, `max 0.051205`

Interpretation:
- Only about `0.8%-5.1%` of the processed tensor is label foreground.
- This is acceptable for a first pass, but it indicates a lot of background is still present.
- The heart itself is much smaller than the processed tensor in `x/y`, so image-based foreground cropping is mainly isolating the torso, not the cardiac region.
- The current training patch size `192 x 192 x 16` appears large enough to contain the observed label boxes with margin, but this should eventually be validated on the full dataset and against validation performance.

## What To Measure Next
- Label bounding-box size after preprocessing across the dataset
- Whether the current training patch size `192 x 192 x 16` covers the heart plus margin in nearly all cases
- Validation metrics from controlled ablations:
  - baseline loading only
  - `+ normalization`
  - `+ normalization + resampling`
  - `+ normalization + resampling + crop/pad`

## Change Policy
Whenever preprocessing parameters change, this file should be updated with:
- what changed
- why it changed
- what evidence motivated the change
- what effect was observed in QC or validation
