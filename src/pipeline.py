from __future__ import annotations

MULTICHANNEL_MODELS = {"25DATTUNET"}
VALID_PREPROCESSING_MODES = {"2d", "2.5d"}


def normalize_preprocessing_mode(mode: str) -> str:
    normalized = mode.strip().lower()
    if normalized not in VALID_PREPROCESSING_MODES:
        available = ", ".join(sorted(VALID_PREPROCESSING_MODES))
        raise ValueError(f"Invalid preprocessing mode '{mode}'. Choose from: {available}")
    return normalized


def preprocessing_mode_uses_triplet_slices(mode: str) -> bool:
    return normalize_preprocessing_mode(mode) == "2.5d"


def model_uses_triplet_slices(model_name: str) -> bool:
    return model_name in MULTICHANNEL_MODELS


def expected_input_channels_for_model(model_name: str) -> int:
    return 3 if model_uses_triplet_slices(model_name) else 1


def validate_model_preprocessing_compatibility(model_name: str, preprocessing_mode: str) -> None:
    model_uses_triplets = model_uses_triplet_slices(model_name)
    mode_uses_triplets = preprocessing_mode_uses_triplet_slices(preprocessing_mode)
    if model_uses_triplets != mode_uses_triplets:
        raise RuntimeError(
            f"MODEL='{model_name}' is incompatible with PREPROCESSING_MODE='{preprocessing_mode}'. "
            f"Model expects {'triplet slices' if model_uses_triplets else 'single slices'}, "
            f"but preprocessing mode produces {'triplet slices' if mode_uses_triplets else 'single slices'}."
        )
