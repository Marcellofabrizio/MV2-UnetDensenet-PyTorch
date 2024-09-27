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

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=torch.cuda.is_available())
val_loader = DataLoader(val_dataset, batch_size=32, num_workers=4, pin_memory=torch.cuda.is_available())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UnetDensenet((224, 224, 1)).to(device)
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 1e-5)
auc_metric = ROCAUCMetric()

post_pred = CreatePrePedictionTransforms()
post_label = CreatePostLabelTransforms()

val_interval = 2
best_metric = -1
best_metric_epoch = -1
writer = SummaryWriter()

for epoch in range(5):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{5}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_len = len(train_dataset) // train_loader.batch_size
        print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
        writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
    epoch_loss /= step
    
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            y_pred = torch.tensor([], dtype=torch.float32, device=device)
            y = torch.tensor([], dtype=torch.long, device=device)
            for val_data in val_loader:
                val_images, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                y_pred = torch.cat([y_pred, model(val_images)], dim=0)
                y = torch.cat([y, val_labels], dim=0)
            acc_value = torch.eq(y_pred.argmax(dim=1), y)
            acc_metric = acc_value.sum().item() / len(acc_value)
            y_onehot = [post_label(i) for i in decollate_batch(y, detach=False)]
            y_pred_act = [post_pred(i) for i in decollate_batch(y_pred)]
            auc_metric(y_pred_act, y_onehot)
            auc_result = auc_metric.aggregate()
            auc_metric.reset()
            del y_pred_act, y_onehot
            if acc_metric > best_metric:
                best_metric = acc_metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), "best_metric_model_classification3d_dict.pth")
                print("saved new best metric model")
            print(
                "current epoch: {} current accuracy: {:.4f} current AUC: {:.4f} best accuracy: {:.4f} at epoch {}".format(
                    epoch + 1, acc_metric, auc_result, best_metric, best_metric_epoch
                )
            )
            writer.add_scalar("val_accuracy", acc_metric, epoch + 1)
print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
writer.close()