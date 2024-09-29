import os
import tempfile
import torch
import numpy as np
import matplotlib.pyplot as plt

# -------------------- MONAI --------------------

from monai.transforms import(
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    BorderPadd,
    Spacingd,
    RandAdjustContrastd,
    RandRotate90d,
    SpatialCropd,
    ToTensord,
    ScaleIntensityd,
    EnsureType,
    EnsureTyped,
    MeanEnsembled,
    Activationsd,
    AsDiscreted,
    SaveImaged,
    ScaleIntensity,
    VoteEnsembled,
    Activations
)

def CreatePrePedictionTransforms():
    return Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=1)])

def CreatePostLabelTransforms():
    return Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold_values=True)])

def CreatePostTransTransforms():
    return Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

def CreateBaseTransforms():
    return Compose([
        LoadImaged(keys=["image", "label"]),
        ScaleIntensityd(keys=["image", "label"], minv=0.0, maxv=1.0),
        EnsureChannelFirstd(keys=["image", "label"]),
        RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=[0, 1])
    ])

def CreateEnsembleTransforms(pred_models):
    keys = ["pred{}".format(i) for i in range(pred_models)]
    return Compose(
        [
            EnsureTyped(keys=["pred0", "pred1", "pred2", "pred3", "pred4"]),
            Activationsd(keys=["pred0", "pred1", "pred2", "pred3", "pred4"], sigmoid=True),
            AsDiscreted(keys=["pred0", "pred1", "pred2", "pred3", "pred4"], threshold=0.5),
            VoteEnsembled(keys=["pred0", "pred1", "pred2", "pred3", "pred4"], output_key="pred"),
            # SaveImaged(keys="pred", 
            #            output_dir="/home/marcello/Repositories/DICOM-Project-Pytorch/out", 
            #            output_postfix="seg", 
            #            output_ext='.png',
            #            resample=False)
        ]
    )

def CreateTrainTransforms(cropSize=[64,64,64], padding=10, num_sample=10):
    return Compose(
        [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
        SpatialCropd(keys=["image", "label"], roi_center=[135,100,111], roi_size=[140,140,111]),
        BorderPadd(keys=["image", "label"],spatial_border=padding),
        ScaleIntensityd(
            keys=["image"],minv = 0.0, maxv = 1.0, factor = None
        ),
        RandShiftIntensityd(
            keys=["image"],
            offsets=0.10,
            prob=0.50,
        ),
        RandAdjustContrastd(
            keys=["image"],
            prob=0.8,
            gamma = (0.5,2)
        ),
        ToTensord(keys=["image", "label"]),
        ]
    )
    
def CreateValidationTransforms():
        return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
            ScaleIntensityd(
                keys=["image"],minv = 0.0, maxv = 1.0, factor = None
            ),
            RandAdjustContrastd(
                keys=["image"],
                prob=0.8,
                gamma = (0.5,2)
            ),
            ToTensord(keys=["image", "label"]),
        ]
    )