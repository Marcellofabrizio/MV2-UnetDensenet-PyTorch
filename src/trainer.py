import torch
from torch.utils.tensorboard import SummaryWriter
from monai.data import (
    decollate_batch,
)

from monai.losses import DiceLoss

from monai.metrics import ROCAUCMetric

from monai.engines import SupervisedEvaluator, SupervisedTrainer, EnsembleEvaluator
from monai.handlers import MeanDice, StatsHandler, ValidationHandler, from_engine
from monai.inferers import SimpleInferer, SlidingWindowInferer
from monai.losses import DiceLoss

from utils import CreatePostLabelTransforms, CreatePrePedictionTransforms
def trainer_module(model, train_loader, train_dataset, val_loader, device=None):
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    loss_function = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True)
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
    
def train(train_loader, val_loader, net, device):
    loss = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True)
    opt = torch.optim.Adam(net.parameters(), 1e-3)

    # val_post_transforms = Compose(
    #     [EnsureTyped(keys="pred"), Activationsd(keys="pred", sigmoid=True), AsDiscreted(keys="pred", threshold=0.5)]
    # )

    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=val_loader,
        network=net,
        # inferer=SlidingWindowInferer(roi_size=(112, 112), sw_batch_size=4, overlap=0.5),
        # postprocessing=val_post_transforms,
        key_val_metric={
            "val_mean_dice": MeanDice(
                include_background=True,
                output_transform=from_engine(["pred", "label"]),
            )
        },
    )
    train_handlers = [
        ValidationHandler(validator=evaluator, interval=4, epoch_level=True),
        StatsHandler(tag_name="train_loss", output_transform=from_engine(["loss"], first=True)),
    ]

    trainer = SupervisedTrainer(
        device=device,
        max_epochs=200,
        train_data_loader=train_loader,
        network=net,
        optimizer=opt,
        loss_function=loss,
        inferer=SimpleInferer(),
        amp=False,
        train_handlers=train_handlers,
    )
    trainer.run()
    return net

def ensemble_evaluate(post_transforms, models, test_loader, device):
    evaluator = EnsembleEvaluator(
        device=device,
        val_data_loader=test_loader,
        pred_keys=["pred{}".format(i) for i in range(len(models))], 
        networks=models,
        postprocessing=post_transforms,
        key_val_metric={
            "test_mean_dice": MeanDice(
                include_background=True,
                output_transform=from_engine(["pred", "label"]),
            )
        },
    )
    evaluator.run()