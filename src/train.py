# acdc/train.py
from src.load_data import build_acdc_list, split_by_patient, build_loaders
from src.transforms import build_train_transform, build_val_transform


def main():
    items = build_acdc_list()
    train_items, val_items = split_by_patient(items, n_splits=5, fold=0)

    train_t = build_train_transform(
        target_spacing=(1.25, 1.25, -1),
        patch_size=(192, 192, 16),
        num_samples=4,
    )
    val_t = build_val_transform(
        target_spacing=(1.25, 1.25, -1),
        pad_size=(192, 192, 16),
    )

    train_loader, val_loader = build_loaders(
        train_items=train_items,
        val_items=val_items,
        train_transform=train_t,
        val_transform=val_t,
        batch_size=2,
        num_workers=4, # number CPU cores
    )

    batch = next(iter(train_loader))
    print("Train batch keys:", batch.keys())
    print("Image shape:", batch["image"].shape)  # (B, C, X, Y, Z)
    print("Label shape:", batch["label"].shape)


if __name__ == "__main__":
    main()
