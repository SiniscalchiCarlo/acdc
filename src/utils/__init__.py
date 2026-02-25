def split_by_patient(
    items: List[Dict[str, str]],
    n_splits: int = 5,
    fold: int = 0,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Split items using GroupKFold by patient id.
    This prevents ED/ES from the same patient leaking across splits.

    Args:
        items: list from build_acdc_list()
        n_splits: number of folds
        fold: which fold to use as validation (0..n_splits-1)

    Returns:
        train_items, val_items
    """
    groups = np.array([it["patient"] for it in items])
    idx = np.arange(len(items))

    gkf = GroupKFold(n_splits=n_splits)
    splits = list(gkf.split(idx, y=None, groups=groups))

    train_idx, val_idx = splits[fold]
    train_items = [items[i] for i in train_idx]
    val_items = [items[i] for i in val_idx]
    return train_items, val_items

