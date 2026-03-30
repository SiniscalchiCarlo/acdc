from __future__ import annotations

from collections.abc import Hashable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from monai.config import KeysCollection
from monai.transforms import (
    Compose,
    DivisiblePadd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    RandAdjustContrastd,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandShiftIntensityd,
    SpatialPadd,
    Spacingd,
    Transform,
)

from config import patch_size_2d, target_spacing_2d

DEFAULT_TARGET_SPACING_2D = target_spacing_2d
DEFAULT_PATCH_SIZE_2D = patch_size_2d

class ExtractSliceTripletd(MapTransform):
    """Extract 3 consecutive slices: [idx-1, idx, idx+1].
    
    Image becomes [3, H, W] (three slices as channels).
    Label stays [1, H, W] — center slice only.
    Edge slices are handled by clamping (repeats boundary slice).
    """

    def __init__(
        self,
        keys: KeysCollection,
        label_keys: KeysCollection,
        index_key: str = "slice_idx",
        allow_missing_keys: bool = False,
    ):
        super().__init__(list(keys) + list(label_keys), allow_missing_keys)
        self.image_keys = list(keys)
        self.label_keys = list(label_keys)
        self.index_key = index_key

    def __call__(self, data: Mapping[Hashable, Any]) -> dict[Hashable, Any]:
        d = dict(data)
        if self.index_key not in d:
            raise KeyError(f"Missing required slice index key: {self.index_key}")

        idx = int(d[self.index_key])

        for key in self.image_keys:
            tensor = d[key]                    # [C, H, W, D]
            n_slices = tensor.shape[-1]
            prev_idx  = max(idx - 1, 0)
            next_idx  = min(idx + 1, n_slices - 1)
            # Stack along channel dim → [3, H, W]
            d[key] = np.concatenate(
                [tensor[..., prev_idx],
                 tensor[..., idx],
                 tensor[..., next_idx]],
                axis=0,
            )

        for key in self.label_keys:
            tensor = d[key]                    # [C, H, W, D]
            d[key] = tensor[..., idx]          # [C, H, W] — center only

        return d


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
            d[key] = tensor[..., slice_idx]

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

def build_preprocessing_transform(
    target_spacing: tuple[float, float, float] = DEFAULT_TARGET_SPACING_2D,
    patch_size: tuple[int, int] = DEFAULT_PATCH_SIZE_2D,
) -> Compose:
    """
    Build deterministic preprocessing for slice-based 2D training.
    This preprocessing is performed in advance to speed up training time.

    The source files remain 3D ACDC volumes, but each dataset item carries a
    `slice_idx` and is converted into a `[C, H, W]` tensor before augmentation.
    """
    transforms: list[Transform] = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"]),
        Spacingd(
            keys=["image", "label"],
            pixdim=target_spacing,
            mode=("bilinear", "nearest"),
        ),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        # Changed: image gets 3 channels, label gets center slice only
        ExtractSliceTripletd(keys=["image"], label_keys=["label"], index_key="slice_idx"),
        SpatialPadd(keys=["image", "label"], spatial_size=patch_size),
        DivisiblePadd(keys=["image", "label"], k=16),
        EnsureTyped(keys=["image", "label"]),
    ]
    return Compose(transforms)


def build_preprocessed_augment_transform(
    patch_size: tuple[int, int] = DEFAULT_PATCH_SIZE_2D,
) -> Compose:
    """Apply online augmentation to already-loaded preprocessed 2D slices."""
    return Compose(
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
            RandShiftIntensityd(keys=["image"], prob=0.5, offsets=0.1),
            RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7, 1.5)),
            RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.01),
            RandGaussianSmoothd(
                keys=["image"],
                prob=0.15,
                sigma_x=(0.5, 1.0),
                sigma_y=(0.5, 1.0),
            ),
            SpatialPadd(keys=["image", "label"], spatial_size=patch_size),
            DivisiblePadd(keys=["image", "label"], k=16),
            EnsureTyped(keys=["image", "label"]),
        ]
    )



# Transforms used in the dataloader (on the already preprocessed data)
# Augmentation (build_preprocessed_augment_transform) is not performed in advance and is applied
# in the build_preprocessed_train_transform
def build_preprocessed_train_transform(
    patch_size: tuple[int, int] = DEFAULT_PATCH_SIZE_2D,
) -> Compose:
    """Load offline-preprocessed 2D slices and apply online augmentation."""
    return Compose(
        [
            LoadPreprocessedSliceD(),
            EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
            EnsureTyped(keys=["image", "label"]),
            build_preprocessed_augment_transform(patch_size=patch_size),
        ]
    )


def build_preprocessed_val_transform(
    patch_size: tuple[int, int] = DEFAULT_PATCH_SIZE_2D,
) -> Compose:
    """Load offline-preprocessed 2D slices and apply final padding."""
    transforms: list[Transform] = [
        LoadPreprocessedSliceD(),
        EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
        EnsureTyped(keys=["image", "label"]),
    ]
    transforms.extend(
        [
            SpatialPadd(keys=["image", "label"], spatial_size=patch_size),
            DivisiblePadd(keys=["image", "label"], k=16),
            EnsureTyped(keys=["image", "label"]),
        ]
    )
    return Compose(transforms)
