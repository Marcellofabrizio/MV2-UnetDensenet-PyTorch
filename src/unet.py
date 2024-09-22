import torch

from torch import nn
from torchvision.models import densenet121, DenseNet121_Weights
from monai.networks.nets.densenet import DenseNet121
from torch.nn import functional as F

class UNetConvBnReLU(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        parametric=False,
        kernel_size=(4, 4), 
    ):
        super(UNetConvBnReLU, self).__init__()
        
        print('UNetConvBnReLU receiving...')
        print('Filters {}, {}'.format(in_channels, out_channels))
        
        self.conv = nn.Conv2d(in_channels=in_channels+1024, 
                              out_channels=out_channels,
                              kernel_size=kernel_size, 
                              stride=1, 
                              padding=1)

        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        print(x.shape)
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class Upsample2DBlock(nn.Module):
    def __init__(self, ch_in, ch_out=None, skip_in=0, parametric=False):
        """
        Upsample2DBlock performs either parametric (ConvTranspose2d) or non-parametric 
        (bilinear interpolation + Conv2d) upsampling, with optional skip connections.
        
        Args:
        - ch_in: Input channel size.
        - ch_out: Output channel size. Defaults to half of ch_in if not provided.
        - skip_in: Additional input channels from the skip connection.
        - parametric: If True, use ConvTranspose2d for upsampling, otherwise use interpolation.
        """
        super(Upsample2DBlock, self).__init__()

        ch_out = ch_in // 2 if ch_out is None else ch_out

        self.parametric = parametric

        if parametric:
            self.up = nn.ConvTranspose2d(ch_in, ch_out, kernel_size=4, stride=2, padding=1)
            self.conv1 = None
            self.bn1 = None
        else:
            self.up = None
            conv1_in = ch_in + skip_in if not parametric else ch_out
            self.conv1 = nn.Conv2d(conv1_in, ch_out, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(ch_out)
            

        conv2_in = ch_out + skip_in if parametric else ch_out
        self.conv2 = nn.Conv2d(conv2_in, ch_out, kernel_size=3, padding=1)

        self.bn2 = nn.BatchNorm2d(ch_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip_connection=None):
        """
        Forward pass of the Upsample2DBlock.
        
        Args:
        - x: Input tensor.
        - skip_connection: Skip connection tensor from an earlier layer, if available.
        
        Returns:
        - x: Output tensor after upsampling and convolutions.
        """
        if self.parametric:
            x = self.up(x)
        else:
            x = F.interpolate(x, size=None, scale_factor=2, mode='bilinear', align_corners=False)

        if skip_connection is not None:
            x = torch.cat([x, skip_connection], dim=1)
        x = self._conv_bn_relu(self.conv1, self.bn1, x)

        x = self._conv_bn_relu(self.conv2, self.bn2, x)
        return x

    def _conv_bn_relu(self, conv, bn, x):
        """
        Helper function to apply convolution, batch normalization (if applicable), and ReLU activation.
        
        Args:
        - conv: Convolution layer to apply.
        - bn: Batch Normalization layer (optional).
        - x: Input tensor.
        
        Returns:
        - x: Output tensor after applying conv, bn, and ReLU.
        """
        
        if conv is not None :
            x = conv(x)
        if bn is not None:
            x = bn(x)
        x = self.relu(x)
        return x

class UnetDensenet(nn.Module):
    def __init__(
        self,
        input_shape,
        growth_rate=32,
        block_config=(6, 12, 24, 16),
        decoder_filters=(256, 128, 64, 32, 16),
        num_init_features=64,
    ):
        super(UnetDensenet, self).__init__()
        
        self.backbone, self.shortcut_features, self.bb_out_name = self.get_backbone()
        shortcut_chs, bb_out_chs = self.infer_skip_channels()

        self.upsample_blocks = nn.ModuleList()
        decoder_filters = decoder_filters[:len(self.shortcut_features)]
        decoder_filters_in = [bb_out_chs] + list(decoder_filters[:-1])
        num_blocks = len(self.shortcut_features)
        
        print('decoder_filters_in {} decoder_filters {}'.format(decoder_filters_in, decoder_filters))
        
        for i, [filters_in, filters_out] in enumerate(zip(decoder_filters_in, decoder_filters)):
            print('upsample_blocks[{}]\nin: {}\nout: {}\n'.format(i, filters_in, filters_out))
            skip_in=shortcut_chs[num_blocks-i-1]
            print('skip_in {}'.format(skip_in))
            self.upsample_blocks.append(Upsample2DBlock(filters_in, filters_out,
                                                      skip_in=skip_in,
                                                      parametric=True))
        
        self.final_conv = nn.Conv2d(decoder_filters[-1], 1, kernel_size=(1, 1))

    def forward(self, *input):
        x, features = self.forward_backbone(*input)

        for skip_name, upsample_block in zip(self.shortcut_features[::-1], self.upsample_blocks):
            skip_features = features[skip_name]
            x = upsample_block(x, skip_features)

        x = self.final_conv(x)
        return x
    
    def forward_backbone(self, x):
        features = {None: None} if None in self.shortcut_features else dict()
        for name, child in self.backbone.named_children():
            x = child(x)
            if name in self.shortcut_features:
                features[name] = x
            if name == self.bb_out_name:
                break

        return x, features

    def get_backbone(self):
        
        backbone = DenseNet121(in_channels=1, out_channels=1, spatial_dims=2, pretrained=True).features
        # backbone = densenet121(weights=DenseNet121_Weights.DEFAULT).features
        feature_names = [None, 'relu0', 'denseblock1', 'denseblock2', 'denseblock3']
        backbone_output = 'denseblock4'
        
        return backbone, feature_names, backbone_output
    
    def infer_skip_channels(self):

        x = torch.zeros(1, 1, 224, 224)
        channels = [0]

        for name, child in self.backbone.named_children():
            x = child(x)
            if name in self.shortcut_features:
                channels.append(x.shape[1])
            if name == self.bb_out_name:
                out_channels = x.shape[1]
                break
        return channels, out_channels

