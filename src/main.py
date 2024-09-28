from training import trainer, trainer_v2
from unet import UnetDensenet
from utils import CreateBaseTransforms, CreatePostLabelTransforms, CreatePrePedictionTransforms

import json

import torch
from torch.utils.tensorboard import SummaryWriter

import monai

from monai.metrics import ROCAUCMetric

from monai.data import (
    DataLoader,
    Dataset, 
    decollate_batch,
    partition_dataset,
)

dataset_path = '/home/marcello/Repositories/DICOM-Project-Pytorch/data/dataset1/dataset1.json'

with open(dataset_path, 'r') as file:
    dataset = json.load(file)

image_paths = [item['image'] for item in dataset['data']]
labels = [item['label'] for item in dataset['data']]

data = list(zip(image_paths, labels))

train_data, val_data, test_data = partition_dataset(
    data, ratios=[0.7, 0.15, 0.15], shuffle=True, seed=42
)

print(train_data[0])

train_images = [{"image": img, "label": label} for img, label in train_data]
val_images = [{"image": img, "label": label} for img, label in val_data]
test_images = [{"image": img, "label": label} for img, label in test_data]

print("{} training images".format(len(train_images)))
print("{} validation images".format(len(val_images)))
print("{} testing images".format(len(test_images)))


train_dataset = Dataset(
    data=train_images,
    transform=CreateBaseTransforms())

val_dataset   = Dataset(
    data=val_images, 
    transform=CreateBaseTransforms())

test_dataset  = Dataset(
    data=test_images, 
    transform=CreateBaseTransforms())

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=2, num_workers=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UnetDensenet((224, 224, 1)).to(device)

trainer_v2(model, train_loader, train_dataset, val_loader, device)