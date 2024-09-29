from cv_dataset import CVDataset
from training import ensemble_evaluate, train
from unet import UnetDensenet
from utils import CreateBaseTransforms, CreateEnsembleTransforms

import matplotlib.pyplot as plt
import numpy as np
import torchvision

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

num = 5
folds = list(range(num))

cvdataset = CrossValidation(
    dataset_cls=CVDataset,
    data=train_val_images,
    nfolds=num,
    seed=12345,
    transform=CreateBaseTransforms(),
)

train_dataset = [cvdataset.get_dataset(folds=folds[0:i] + folds[(i + 1) :]) for i in folds]

val_dataset = [cvdataset.get_dataset(folds=i, transform=CreateBaseTransforms()) for i in range(num)]

test_dataset  = Dataset(
    data=test_images, 
    transform=CreateBaseTransforms())

train_loaders = [DataLoader(train_dataset[i], batch_size=1, shuffle=True, num_workers=4, pin_memory=False) for i in folds]
val_loaders = [DataLoader(val_dataset[i], batch_size=1, num_workers=4, pin_memory=False) for i in folds]
test_loader = DataLoader(test_dataset, batch_size=1, num_workers=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UnetDensenet((224, 224, 1)).to(device)

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

images = 0
imgs = []

for i in train_loaders[0]:
    if images == 4:
        break
    
    images += 1
    print(i['image'].shape)
    imgs.append(i['image'])

npimg = torch.squeeze(imgs[0], dim=1).squeeze(0).numpy()
plt.imshow(npimg,cmap='gray')
plt.show()


models = [train(train_loaders[i], val_loaders[i], model, device) for i in range(num)]

mean_post_transforms = CreateEnsembleTransforms(len(models))

print(ensemble_evaluate(mean_post_transforms, models, test_loader, device))