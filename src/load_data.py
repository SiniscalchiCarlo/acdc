from pathlib import Path
import re
from config import dataset_path
from typing import List, Dict, Tuple
import numpy as np
from sklearn.model_selection import GroupKFold
from monai.data import CacheDataset, DataLoader
from monai.utils import set_determinism

def parse_info_cfg(info_path: Path):
    """
    Parse Info.cfg ACDC:
    extracts ED e ES (frame indices).
    """
    txt = info_path.read_text(errors="ignore")
    m_ed = re.search(r"\bED\s*:\s*(\d+)", txt)
    m_es = re.search(r"\bES\s*:\s*(\d+)", txt)
    ed = int(m_ed.group(1)) if m_ed else None
    es = int(m_es.group(1)) if m_es else None
    return ed, es

def build_acdc_list():
    root = Path(dataset_path)
    patients = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("patient")])
    items = []
    for p in patients:
        ed, es = parse_info_cfg(p / "Info.cfg")
        for fr, phase in [(ed, "ED"), (es, "ES")]:
            img = p / f"{p.name}_frame{fr:02d}.nii.gz"
            lab = p / f"{p.name}_frame{fr:02d}_gt.nii.gz"
            items.append({"image": str(img), "label": str(lab), "patient": p.name, "phase": phase})
    return items

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

def build_loaders(
    train_items: List[Dict[str, str]],
    val_items: List[Dict[str, str]],
    train_transform,
    val_transform,
    batch_size: int = 2,
    num_workers: int = 4,
    cache_rate_train: float = 0.5,
    cache_rate_val: float = 1.0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create MONAI CacheDataset + DataLoaders.

    Notes:
      - CacheDataset speeds up training significantly.
      - Consider adjusting cache_rate based on available RAM.
      - set_determinism helps reproducibility.
    """
    set_determinism(seed=seed)

    train_ds = CacheDataset(
        data=train_items,
        transform=train_transform,
        cache_rate=cache_rate_train,
        num_workers=num_workers,
    )
    val_ds = CacheDataset(
        data=val_items,
        transform=val_transform,
        cache_rate=cache_rate_val,
        num_workers=num_workers,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        # Respect an explicit request for single-process loading. This keeps the
        # baseline runnable in restricted environments and makes debugging easier.
        num_workers=max(0, num_workers // 2),
        pin_memory=True,
    )
    return train_loader, val_loader
