# acdc/transforms/monai_transforms.py
from __future__ import annotations
from monai.transforms import RandFlipd

import numpy as np
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    Orientationd,
    Spacingd,
    NormalizeIntensityd,
    CropForegroundd,
    SpatialPadd,
    RandCropByPosNegLabeld,
    RandAffined,
    Rand3DElasticd,
    RandShiftIntensityd,
    RandAdjustContrastd,
    RandGaussianNoised,
    RandGaussianSmoothd
)
from typing import Tuple

DEFAULT_TARGET_SPACING = (1.25, 1.25, -1.0)


def build_preprocess_transform(
    target_spacing: Tuple[float, float, float] = DEFAULT_TARGET_SPACING,
    pad_size: Tuple[int, int, int] = (192, 192, 16),
) -> Compose:
    """
    Build deterministic preprocessing:
      - load NIfTI
      - channel-first
      - unify orientation
      - resample to target spacing (semi-isotropic)
      - per-volume z-score normalization
      - foreground crop to remove large background
      - pad to guarantee minimum size

    Notes:
      - Use bilinear interpolation for images and nearest for labels.
      - Keep native z spacing with -1.0 because ACDC slice thickness is much
        larger than the in-plane spacing, so forcing isotropy would add
        through-plane interpolation artifacts.
      - nonzero=True is important for MRI to avoid background bias in mean/std.
    """
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            # Ensure input data to be pyTorch tensor or np array
            EnsureTyped(keys=["image", "label"]),

            # Enforce a common orientation so downstream spatial operations
            # behave consistently across scanners and sites.
            Orientationd(keys=["image", "label"], axcodes="RAS"),

            # Bilinear interpolation is appropriate for continuous MR intensities,
            # while nearest-neighbor keeps label ids discrete.
            Spacingd(
                keys=["image", "label"],
                pixdim=target_spacing,
                mode=("bilinear", "nearest"),
            ),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            SpatialPadd(keys=["image", "label"], spatial_size=pad_size),
        ]
    )


def build_train_transform(
    target_spacing: Tuple[float, float, float] = DEFAULT_TARGET_SPACING,
    patch_size: Tuple[int, int, int] = (192, 192, 16),
    num_samples: int = 4,
) -> Compose:
    """
    Build training transform:
      preprocess -> label-guided ROI sampling -> heavy but realistic augmentation

    Label-guided cropping:
      RandCropByPosNegLabeld samples patches that contain foreground (heart) and background,
      improving training stability when the heart occupies a small fraction of the volume.
    """
    preprocess = build_preprocess_transform(target_spacing=target_spacing, pad_size=patch_size)

    roi_sampler = RandCropByPosNegLabeld(
        keys=["image", "label"],
        label_key="label",
        spatial_size=patch_size,
        pos=1,
        neg=1,
        num_samples=num_samples,
        image_key="image",
        image_threshold=0,
    )

    augment = Compose(
        [
            RandAffined(
                keys=["image", "label"],
                prob=0.5,
                # Keep rotations moderate for cardiac SA stacks
                rotate_range=(0.0, 0.0, np.deg2rad(15.0)),
                # Scale mostly in-plane; z scaling often less meaningful for thick slices
                scale_range=(0.10, 0.10, 0.0),
                translate_range=(5, 5, 0),
                mode=("bilinear", "nearest"),
                padding_mode="border",
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            #Rand3DElasticd(
            #    keys=["image", "label"],
            #    prob=0.25,
            #    sigma_range=(3, 5),
            #    magnitude_range=(30, 60),
            #    mode=("bilinear", "nearest"),
            #    padding_mode="border",
            #),
            RandShiftIntensityd(keys=["image"], prob=0.5, offsets=0.1),
            # Contrast/gamma-like adjustment (often used as a proxy for gamma correction)
            RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.8, 1.2)),
            RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.01),
            RandGaussianSmoothd(
                keys=["image"],
                prob=0.15,
                sigma_x=(0.5, 1.0),
                sigma_y=(0.5, 1.0),
                sigma_z=(0.0, 0.0),
            ),
        ]
    )

    return Compose([preprocess, roi_sampler, augment])


def build_val_transform(
    target_spacing: Tuple[float, float, float] = DEFAULT_TARGET_SPACING,
    pad_size: Tuple[int, int, int] = (192, 192, 16),
) -> Compose:
    """
    Validation transform:
      Use deterministic preprocessing only.
    """
    return build_preprocess_transform(target_spacing=target_spacing, pad_size=pad_size)
