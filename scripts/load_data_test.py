from monai.data.dataset import CacheDataset
from src.load_data import build_acdc_list
from transforms import build_train_transform

train_items = build_acdc_list()
train_transform = build_train_transform() 

# train_ds = CacheDataset(
#     data=train_items,
#     transform=train_transform,
#     cache_rate=cache_rate_train,
#     num_workers=num_workers,
# )

