import torch

from torch import nn


class UnetDensenet(nn.Module):
    def __init__(
        self,
        n_classes,
        growth_rate=32,
        block_config=(6, 12, 24, 16),
        num_init_features=64,
        bn_size=4,
        drop_rate=0,
        downsample=False,
        pretrained_encoder_uri=None,
        progress=None,
    ):
        super(UnetDensenet, self).__init__()
