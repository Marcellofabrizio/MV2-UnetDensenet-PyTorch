from utils import *

from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
)

import matplotlib.pyplot as plt

train_transforms = CreateTrainTransforms()
val_transforms = CreateValidationTransforms()

data_dir = "data/dataset0/"
split_json = "dataset_0.json"

datasets = data_dir + split_json
datalist = load_decathlon_datalist(datasets, True, "training")
val_files = load_decathlon_datalist(datasets, True, "validation")
train_ds = CacheDataset(
    data=datalist,
    transform=train_transforms,
    cache_num=24,
    cache_rate=1.0,
    num_workers=8,
)
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)

slice_map = {
    "coleta_3.nii.gz": 78,
}

case_num = 0

img_name = os.path.split(train_ds[case_num]["image"].meta["filename_or_obj"])[1]
img = train_ds[case_num]["image"]
label = train_ds[case_num]["label"]
img_shape = img.shape
label_shape = label.shape
print(f"image shape: {img_shape}, label shape: {label_shape}")
plt.figure("image", (18, 6))
plt.subplot(1, 2, 1)
plt.title("image")
plt.imshow(img[0, :, :, slice_map[img_name]].detach().cpu(), cmap="gray")
plt.subplot(1, 2, 2)
plt.title("label")
plt.imshow(label[0, :, :, slice_map[img_name]].detach().cpu())
plt.show()