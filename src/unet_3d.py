import torch

from torch import nn
from torchvision.models import densenet121


class _UpsampleBlock(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels=None, 
        kernel_size=(3,3,3),
        padding=(0, 0), 
        size=(2,2)
    ):
        super(_UpsampleBlock, self).__init__()

        out_channels = in_channels // 2 if out_channels is None else out_channels

        self.upsample = nn.Upsample(
            scale_factor=size, mode="trilinear", align_corners=True
        )
        self.conv = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=(kernel_size, 3, 3),
                padding=padding,
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x, projected_residual):
        """
        :param x:  image tensor
        :return:   output of the forward pass
        """
        residual = torch.cat(
            (self.upsample(x), self.upsample(projected_residual)),
            dim=1,
        )
        return self.conv(residual)

class UnetDensenet(nn.Module):
    def __init__(
        self,
        input_shape,
        growth_rate=32,
        block_config=(6, 12, 24, 16),
        num_init_features=64,
    ):
        super(UnetDensenet, self).__init__()

    def get_backbone(self):
        return densenet121(pretrained=True).features
