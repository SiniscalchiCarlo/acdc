from __future__ import annotations

from collections.abc import Hashable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from monai.config import KeysCollection
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandAdjustContrastd,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandShiftIntensityd,
    ResizeWithPadOrCropd,
    SpatialPadd,
    Spacingd,
    Transform,
)

DEFAULT_TARGET_SPACING_2D = (1.25, 1.25, -1.0)
DEFAULT_PATCH_SIZE_2D = (192, 192)
DEFAULT_FOREGROUND_MARGIN_2D = 16


class ExtractSliceByIndexd(MapTransform):
    """Extract one 2D slice from a 3D volume using the sample `slice_idx` metadata."""

    def __init__(self, keys: KeysCollection, index_key: str = "slice_idx", allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.index_key = index_key

    def __call__(self, data: Mapping[Hashable, Any]) -> dict[Hashable, Any]:
        d = dict(data)
        if self.index_key not in d:
            raise KeyError(f"Missing required slice index key: {self.index_key}")

        slice_idx = int(d[self.index_key])
        for key in self.key_iterator(d):
            tensor = d[key]
            if tensor.ndim != 4:
                raise ValueError(f"Expected [C, H, W, D] tensor for key '{key}', got shape {tuple(tensor.shape)}")

            depth = int(tensor.shape[-1])
            if depth < 1:
                raise ValueError(f"Cannot extract a slice from empty depth for key '{key}'")

            clamped_idx = max(0, min(slice_idx, depth - 1))
            d[key] = tensor[..., clamped_idx]

        return d


class LoadPreprocessedSliceD(Transform):
    """Load one offline-preprocessed 2D sample from a `.npz` file."""

    def __init__(self, sample_key: str = "sample"):
        self.sample_key = sample_key

    def __call__(self, data: Mapping[Hashable, Any]) -> dict[Hashable, Any]:
        d = dict(data)
        sample_path = Path(d[self.sample_key])
        with np.load(sample_path, allow_pickle=False) as sample:
            d["image"] = sample["image"].astype(np.float32, copy=False)
            d["label"] = sample["label"].astype(np.int64, copy=False)
        return d


def build_preprocess_transform(
    target_spacing: tuple[float, float, float] = DEFAULT_TARGET_SPACING_2D,
    patch_size: tuple[int, int] = DEFAULT_PATCH_SIZE_2D,
    foreground_margin: int = DEFAULT_FOREGROUND_MARGIN_2D,
) -> Compose:
    """
    Build deterministic preprocessing for slice-based 2D training.

    The source files remain 3D ACDC volumes, but each dataset item carries a
    `slice_idx` and is converted into a `[C, H, W]` tensor before augmentation.
    """
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            EnsureTyped(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=target_spacing,
                mode=("bilinear", "nearest"),
            ),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            ExtractSliceByIndexd(keys=["image", "label"], index_key="slice_idx"),
            CropForegroundd(
                keys=["image", "label"],
                source_key="image",
                margin=foreground_margin,
                allow_smaller=True,
            ),
            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=patch_size),
            EnsureTyped(keys=["image", "label"]),
        ]
    )


def build_preprocess_transform_no_crop(
    target_spacing: tuple[float, float, float] = DEFAULT_TARGET_SPACING_2D,
    pad_size: tuple[int, int] = DEFAULT_PATCH_SIZE_2D,
) -> Compose:
    """Build deterministic preprocessing without foreground cropping."""
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            EnsureTyped(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=target_spacing,
                mode=("bilinear", "nearest"),
            ),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            ExtractSliceByIndexd(keys=["image", "label"], index_key="slice_idx"),
            SpatialPadd(keys=["image", "label"], spatial_size=pad_size),
            EnsureTyped(keys=["image", "label"]),
        ]
    )


def build_train_transform(
    target_spacing: tuple[float, float, float] = DEFAULT_TARGET_SPACING_2D,
    patch_size: tuple[int, int] = DEFAULT_PATCH_SIZE_2D,
    foreground_margin: int = DEFAULT_FOREGROUND_MARGIN_2D,
) -> Compose:
    """Build 2D training transforms that output `[C, H, W]` tensors for a UNet."""

    preprocess = build_preprocess_transform_no_crop(
        target_spacing=target_spacing,
        patch_size=patch_size,
        foreground_margin=foreground_margin,
    )

    augment = Compose(
        [
            RandAffined(
                keys=["image", "label"],
                prob=0.7,
                rotate_range=np.deg2rad(20.0),
                scale_range=(0.10, 0.10),
                translate_range=(12, 12),
                shear_range=(0.05, 0.05),
                mode=("bilinear", "nearest"),
                padding_mode="border",
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandShiftIntensityd(keys=["image"], prob=0.5, offsets=0.1),
            RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7, 1.5)),
            RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.01),
            RandGaussianSmoothd(
                keys=["image"],
                prob=0.15,
                sigma_x=(0.5, 1.0),
                sigma_y=(0.5, 1.0),
            ),
            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=patch_size),
            EnsureTyped(keys=["image", "label"]),
        ]
    )
    return Compose([preprocess, augment])


def build_val_transform(
    target_spacing: tuple[float, float, float] = DEFAULT_TARGET_SPACING_2D,
    patch_size: tuple[int, int] = DEFAULT_PATCH_SIZE_2D,
    foreground_margin: int = DEFAULT_FOREGROUND_MARGIN_2D,
) -> Compose:
    """Build deterministic validation transforms for slice-based 2D inference."""
    return build_preprocess_transform_no_crop(
        target_spacing=target_spacing,
        patch_size=patch_size,
        foreground_margin=foreground_margin,
    )


def build_val_transform_no_crop(
    target_spacing: tuple[float, float, float] = DEFAULT_TARGET_SPACING_2D,
    pad_size: tuple[int, int] = DEFAULT_PATCH_SIZE_2D,
) -> Compose:
    """Build deterministic validation preprocessing without foreground cropping."""
    return build_preprocess_transform_no_crop(
        target_spacing=target_spacing,
        pad_size=pad_size,
    )


def build_preprocessed_train_transform(
    patch_size: tuple[int, int] = DEFAULT_PATCH_SIZE_2D,
) -> Compose:
    """Load offline-preprocessed 2D slices and apply augmentation only."""
    return Compose(
        [
            LoadPreprocessedSliceD(),
            EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
            EnsureTyped(keys=["image", "label"]),
            RandAffined(
                keys=["image", "label"],
                prob=0.7,
                rotate_range=np.deg2rad(20.0),
                scale_range=(0.10, 0.10),
                translate_range=(12, 12),
                shear_range=(0.05, 0.05),
                mode=("bilinear", "nearest"),
                padding_mode="border",
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandShiftIntensityd(keys=["image"], prob=0.5, offsets=0.1),
            RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7, 1.5)),
            RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.01),
            RandGaussianSmoothd(
                keys=["image"],
                prob=0.15,
                sigma_x=(0.5, 1.0),
                sigma_y=(0.5, 1.0),
            ),
            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=patch_size),
            EnsureTyped(keys=["image", "label"]),
        ]
    )


def build_preprocessed_val_transform(
    patch_size: tuple[int, int] = DEFAULT_PATCH_SIZE_2D,
) -> Compose:
    """Load offline-preprocessed 2D slices with no random augmentation."""
    return Compose(
        [
            LoadPreprocessedSliceD(),
            EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=patch_size),
            EnsureTyped(keys=["image", "label"]),
        ]
    )
