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
    ToTensord,
    ScaleIntensityd,
)

def CreateTrainTransforms(cropSize=[64,64,64], padding=10, num_sample=10):
    return Compose(
        [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
        BorderPadd(keys=["image", "label"],spatial_border=padding),
        ScaleIntensityd(
            keys=["image"],minv = 0.0, maxv = 1.0, factor = None
        ),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=cropSize,
            pos=1,
            neg=1,
            num_samples=num_sample,
            image_key="image",
            image_threshold=0,
        ),
        # RandFlipd(
        #     keys=["image", "label"],
        #     spatial_axis=[0],
        #     prob=0.20,
        # ),
        # RandFlipd(
        #     keys=["image", "label"],
        #     spatial_axis=[1],
        #     prob=0.20,
        # ),
        # RandFlipd(
        #     keys=["image", "label"],
        #     spatial_axis=[2],
        #     prob=0.20,
        # ),
        # RandRotate90d(
        #     keys=["image", "label"],
        #     prob=0.10,
        #     max_k=3,
        # ),
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
            # CropForegroundd(keys=["image", "label"], source_key="image"),
            # RandFlipd(
            #     keys=["image", "label"],
            #     spatial_axis=[0],
            #     prob=0.20,
            # ),
            # RandFlipd(
            #     keys=["image", "label"],
            #     spatial_axis=[1],
            #     prob=0.20,
            # ),
            # RandFlipd(
            #     keys=["image", "label"],
            #     spatial_axis=[2],
            #     prob=0.20,
            # ),
            # RandRotate90d(
            #     keys=["image", "label"],
            #     prob=0.10,
            #     max_k=3,
            # ),
            RandAdjustContrastd(
                keys=["image"],
                prob=0.8,
                gamma = (0.5,2)
            ),
            ToTensord(keys=["image", "label"]),
        ]
    )