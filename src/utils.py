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
)

def CreatePrePedictionTransforms():
    return Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=2)])

def CreatePostLabelTransforms():
    return Compose([EnsureType(), AsDiscrete(to_onehot=2)])

def CreateBaseTransforms():
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=[0, 1])
    ])

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