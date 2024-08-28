import torch

from torch import nn
from torchvision.models import DenseNet


class _UpsampleBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels=None, skip_in=0, use_bn=True, parametric=False
    ):
        super(_UpsampleBlock, self).__init__()

        out_channels = in_channels / 2 if out_channels is None else out_channels

        self.up = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(4, 4),
            stride=2,
            padding=1,
            output_padding=0,
            bias=(not use_bn),
        )
        self.bn1 = nn.BatchNorm2d(out_channels) if use_bn else None


class UnetDensenet(nn.Module):
    def __init__(
        self,
        input_shape,
        growth_rate=32,
        block_config=(6, 12, 24, 16),
        num_init_features=64,
    ):
        super(UnetDensenet, self).__init__()
