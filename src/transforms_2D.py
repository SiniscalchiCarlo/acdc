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

from config import PATCH_SIZE, PREPROCESSING_MODE as DEFAULT_PREPROCESSING_MODE, TARGET_SPACING
from src.mode_compatibility import preprocessing_mode_uses_triplet_slices

DEFAULT_TARGET_SPACING_2D = TARGET_SPACING
DEFAULT_PATCH_SIZE_2D = PATCH_SIZE

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
        idx = int(d[self.index_key])

        # 1. Handle Images: Stack 3 slices to create 2.5D input [3, H, W]
        for key in self.image_keys:
            tensor = d[key]                    # Expected [C, H, W, D]
            n_slices = tensor.shape[-1]
            prev_idx = max(idx - 1, 0)
            next_idx = min(idx + 1, n_slices - 1)
            
            # Use np.stack to combine the center slice with its neighbors
            # We take the first channel [0] assuming grayscale input
            d[key] = np.stack(
                [tensor[0, ..., prev_idx], 
                tensor[0, ..., idx], 
                tensor[0, ..., next_idx]],
                axis=0,
            )

        # 2. Handle Labels: Extract only the center slice [1, H, W]
        for key in self.label_keys:
            tensor = d[key]                    # [C, H, W, D]
            # Slicing with idx:idx+1 keeps the channel dimension intact
            d[key] = tensor[..., idx]
            if d[key].ndim == 2:               # Safety check for squeezed dims
                d[key] = d[key][None, ...]

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
    """Load one offline-preprocessed slice sample from a `.npz` file."""

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
    preprocessing_mode: str = DEFAULT_PREPROCESSING_MODE,
) -> Compose:
    """
    Build deterministic preprocessing for slice-based training.
    This preprocessing is performed in advance to speed up training time.

    The source files remain 3D ACDC volumes, but each dataset item carries a
    `slice_idx` and is converted into a `[C, H, W]` tensor before augmentation.
    The number of channels depends on `preprocessing_mode`:
    - `2d` -> one slice channel
    - `2.5d` -> three adjacent slices stacked as channels
    """
    if preprocessing_mode_uses_triplet_slices(preprocessing_mode):
        slice_extractor: Transform = ExtractSliceTripletd(keys=["image"], label_keys=["label"], index_key="slice_idx")
    else:
        slice_extractor = Compose(
            [
                ExtractSliceByIndexd(keys=["image"], index_key="slice_idx"),
                ExtractSliceByIndexd(keys=["label"], index_key="slice_idx"),
            ]
        )

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
        slice_extractor,
    ]
    transforms.extend(
        [
            SpatialPadd(keys=["image", "label"], spatial_size=patch_size),
            DivisiblePadd(keys=["image", "label"], k=16),
            EnsureTyped(keys=["image", "label"]),
        ]
    )
    return Compose(transforms)   


def build_preprocessed_augment_transform(
    patch_size: tuple[int, int] = DEFAULT_PATCH_SIZE_2D,
) -> Compose:
    """Apply online augmentation to already-loaded preprocessed slice samples."""
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
    """Load offline-preprocessed slice samples and apply online augmentation.
    
    LoadPreprocessedSliceD already returns image [C, H, W] and label [1, H, W],
    so EnsureChannelFirstd must NOT be used here.
    """
    return Compose(
        [
            LoadPreprocessedSliceD(),
            EnsureTyped(keys=["image", "label"]),
            build_preprocessed_augment_transform(patch_size=patch_size),
        ]
    )


def build_preprocessed_val_transform(
    patch_size: tuple[int, int] = DEFAULT_PATCH_SIZE_2D,
) -> Compose:
    """Load offline-preprocessed slice samples and apply final padding.
    
    LoadPreprocessedSliceD already returns image [C, H, W] and label [1, H, W],
    so EnsureChannelFirstd must NOT be used here.
    """
    return Compose(
        [
            LoadPreprocessedSliceD(),
            EnsureTyped(keys=["image", "label"]),
            SpatialPadd(keys=["image", "label"], spatial_size=patch_size),
            DivisiblePadd(keys=["image", "label"], k=16),
            EnsureTyped(keys=["image", "label"]),
        ]
    )
