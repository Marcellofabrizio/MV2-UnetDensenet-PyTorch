from cv_dataset import CVDataset
from trainer import train
from unet import UnetDensenet
from utils import CreateBaseTransforms

import json

import torch

from monai.apps import (
    CrossValidation
)

from monai.data import (
    DataLoader,
    Dataset, 
    partition_dataset,
)

dataset_path = '/home/marcello/Repositories/DICOM-Project-Pytorch/data/dataset1/dataset1.json'

with open(dataset_path, 'r') as file:
    dataset = json.load(file)

image_paths = [item['image'] for item in dataset['data']]
labels = [item['label'] for item in dataset['data']]

data = list(zip(image_paths, labels))

data_list, test_data = partition_dataset(
    data, ratios=[0.85, 0.15], shuffle=True, seed=42
)

print(data_list[0])

train_val_images = [{"image": img, "label": label} for img, label in data_list]
test_images = [{"image": img, "label": label} for img, label in test_data]

num = 2
folds = list(range(num))

cvdataset = CrossValidation(
    dataset_cls=CVDataset,
    data=train_val_images,
    nfolds=5,
    seed=12345,
    transform=CreateBaseTransforms(),
)

train_dataset = [cvdataset.get_dataset(folds=folds[0:i] + folds[(i + 1) :]) for i in folds]

val_dataset = [cvdataset.get_dataset(folds=i, transform=CreateBaseTransforms()) for i in range(num)]

test_dataset  = Dataset(
    data=test_images, 
    transform=CreateBaseTransforms())

train_loaders = [DataLoader(train_dataset[i], batch_size=2, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available()) for i in folds]
val_loaders = [DataLoader(val_dataset[i], batch_size=1, num_workers=4, pin_memory=torch.cuda.is_available()) for i in folds]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UnetDensenet((224, 224, 1)).to(device)

models = [train(i, train_loaders[i], val_loaders[i], model, device) for i in range(num)]