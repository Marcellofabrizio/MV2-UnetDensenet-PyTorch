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
        strides=(1, 1), 
        padding='same',
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
    def __init__(self, in_channels, out_channels, skip_in=0, kernel_size=(3,3), upsample_rate=(2,2)):
        super(Upsample2DBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=upsample_rate, mode='bilinear', align_corners=True)
        
        self.conv1 = UNetConvBnReLU(in_channels, out_channels, parametric=True, kernel_size=kernel_size)
        self.conv2 = UNetConvBnReLU(in_channels, out_channels+skip_in, kernel_size=kernel_size)

    def forward(self, x, skip=None):
        x = self.upsample(x)
        
        if skip is not None:
            print('Concatenating {} with {}'.format(x.shape, skip.shape))
            x = torch.cat([x, skip], dim=1)
        
        print('Size after concat {}'.format(x.shape))
        
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class UpsampleBlock(nn.Module):

    # TODO: separate parametric and non-parametric classes?
    # TODO: skip connection concatenated OR added

    def __init__(self, ch_in, ch_out=None, skip_in=0, use_bn=True, parametric=False):
        super(UpsampleBlock, self).__init__()

        self.parametric = parametric
        ch_out = ch_in/2 if ch_out is None else ch_out

        # first convolution: either transposed conv, or conv following the skip connection
        if parametric:
            # versions: kernel=4 padding=1, kernel=2 padding=0
            self.up = nn.ConvTranspose2d(in_channels=ch_in, out_channels=ch_out, kernel_size=(4, 4),
                                         stride=2, padding=1, output_padding=0, bias=(not use_bn))
            self.bn1 = nn.BatchNorm2d(ch_out) if use_bn else None
        else:
            self.up = None
            ch_in = ch_in + skip_in
            self.conv1 = nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=(3, 3),
                                   stride=1, padding=1, bias=(not use_bn))
            self.bn1 = nn.BatchNorm2d(ch_out) if use_bn else None

        self.relu = nn.ReLU(inplace=True)

        # second convolution
        conv2_in = ch_out if not parametric else ch_out + skip_in
        self.conv2 = nn.Conv2d(in_channels=conv2_in, out_channels=ch_out, kernel_size=(3, 3),
                               stride=1, padding=1, bias=(not use_bn))
        self.bn2 = nn.BatchNorm2d(ch_out) if use_bn else None

    def forward(self, x, skip_connection=None):

        x = self.up(x) if self.parametric else F.interpolate(x, size=None, scale_factor=2, mode='bilinear',
                                                             align_corners=None)
        if self.parametric:
            x = self.bn1(x) if self.bn1 is not None else x
            x = self.relu(x)

        if skip_connection is not None:
            x = torch.cat([x, skip_connection], dim=1)

        if not self.parametric:
            x = self.conv1(x)
            x = self.bn1(x) if self.bn1 is not None else x
            x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x) if self.bn2 is not None else x
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
        
        print('Shortcut features: {}\nBackbone output names: {}'.format(self.shortcut_features, self.bb_out_name))
        
        shortcut_chs, bb_out_chs = self.infer_skip_channels()

        print('Shortcut Channels: {}\nBackbone output channels: {}'.format(shortcut_chs, bb_out_chs))

        filters = [1024, 512, 256, 128, 64]

        self.upsample_blocks = nn.ModuleList()
        decoder_filters = decoder_filters[:len(self.shortcut_features)]
        decoder_filters_in = [bb_out_chs] + list(decoder_filters[:-1])
        num_blocks = len(self.shortcut_features)
        
        print('decoder_filters_in {} decoder_filters {}'.format(decoder_filters_in, decoder_filters))
        
        for i, [filters_in, filters_out] in enumerate(zip(decoder_filters_in, decoder_filters)):
            print('upsample_blocks[{}]\nin: {}\nout: {}\n'.format(i, filters_in, filters_out))
            skip_in=shortcut_chs[num_blocks-i-1]
            print('skip_in {}'.format(skip_in))
            # self.upsample_blocks.append(Upsample2DBlock(filters_in, filters_out))
            self.upsample_blocks.append(UpsampleBlock(filters_in, filters_out,
                                                      skip_in=skip_in,
                                                      parametric=True,
                                                      use_bn=True))

        
        self.final_conv = nn.Conv2d(decoder_filters[-1], 1, kernel_size=(3, 3))

    def forward(self, *input):

        x, features = self.forward_backbone(*input)

        for skip_name, upsample_block in zip(self.shortcut_features[::-1], self.upsample_blocks):
            skip_features = features[skip_name]
            print('X Shape: {}'.format(x.shape))
            
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

