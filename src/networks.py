import os

import torch
import torch.nn as nn
from libs.Modules import PartialModule, GatedModule, DeConvGatedModule, PartialResnetBlock, CFSModule, ResCFSBlock, \
    DeformableConv2d
import torch.nn.functional as F
from torchvision.ops import deform_conv2d


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.001):
        # def init_weights(self, init_type='xavier', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0
        /models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, std=gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)


class Discriminator(BaseNetwork):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid
        inplace_flag = True
        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=inplace_flag),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=inplace_flag),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=inplace_flag),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=inplace_flag),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]


# Basic Block of residual network
class ResnetBlock(nn.Module):
    def __init__(self, in_channels, dilation=1, use_spectral_norm=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ZeroPad2d(dilation),
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3,
                                    padding=0, dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),
            # nn.LeakyReLU(0.2, inplace=False),
            nn.Tanh(),
            nn.ZeroPad2d(1),
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3,
                                    padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm),
        )

    def forward(self, x):
        out = x + self.conv_block(x)
        # out = nn.LeakyReLU(0.2, inplace=False)(out)
        out = nn.Tanh()(out)
        return out


class DResnetBlock(nn.Module):
    def __init__(self, in_channels, dilation=1, use_spectral_norm=False):
        super(DResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            # nn.ZeroPad2d(dilation),
            spectral_norm(DeformableConv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 3),
                                           padding=1, bias=not use_spectral_norm), use_spectral_norm),
            # nn.LeakyReLU(0.2, inplace=False),
            nn.Tanh(),
            # nn.ZeroPad2d(1),
            spectral_norm(DeformableConv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 3),
                                           padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )

    def forward(self, x):
        out = x + self.conv_block(x)
        # out = nn.LeakyReLU(0.2, inplace=False)(out)
        out = nn.Tanh()(out)
        return out


class BottleneckBlock(nn.Module):
    expansion = 2

    def __init__(self, in_channels, out_channels, stride=1, dilation=1, use_spectral_norm=False, downsample=None):
        super(BottleneckBlock, self).__init__()
        self.downsample = downsample
        self.stride = stride
        self.conv_block = nn.Sequential(
            nn.ZeroPad2d(dilation),
            spectral_norm(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels * 2, kernel_size=3, stride=stride,
                          padding=0, dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),
            # nn.LeakyReLU(0.2, inplace=False),
            nn.Tanh(),
            nn.ZeroPad2d(1),
            spectral_norm(
                nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels, kernel_size=3, stride=stride,
                          padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm),
        )

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.conv_block(x) + residual
        # out = nn.LeakyReLU(0.2, inplace=False)(out)
        out = nn.Tanh()(out)
        return out


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module


class UnetGenerator(BaseNetwork):
    def __init__(self, residual_blocks=8, init_weights=True):
        super(UnetGenerator, self).__init__()
        inplace_flag = True
        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=0),
            # DeformableConv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), padding=0),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(inplace_flag),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            # DeformableConv2d(in_channels=64, out_channels=128, kernel_size=(4, 4), stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(inplace_flag),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            # DeformableConv2d(in_channels=128, out_channels=256, kernel_size=(4, 4), stride=2, padding=1),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(inplace_flag)
        )

        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(256, 2)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(inplace_flag),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(inplace_flag),

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding=0),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = (torch.tanh(x) + 1) / 2

        return x


class DUnetGenerator(BaseNetwork):
    def __init__(self, residual_blocks=8, init_weights=True):
        super(DUnetGenerator, self).__init__()
        inplace_flag = True
        self.encoder = nn.Sequential(
            # nn.ReflectionPad2d(3),
            # nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=0),
            DeformableConv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), padding=3),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(inplace_flag),

            # nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            DeformableConv2d(in_channels=64, out_channels=128, kernel_size=(4, 4), stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(inplace_flag),

            # nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            DeformableConv2d(in_channels=128, out_channels=256, kernel_size=(4, 4), stride=2, padding=1),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(inplace_flag)
        )

        blocks = []
        for _ in range(residual_blocks):
            block = DResnetBlock(256, 2)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(inplace_flag),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(inplace_flag),

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding=0),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = (torch.tanh(x) + 1) / 2

        return x


class DUnetLink(BaseNetwork):
    def __init__(self, residual_blocks=8, init_weights=True):
        super(DUnetLink, self).__init__()
        inplace_flag = True
        self.encoder1 = nn.Sequential(
            # nn.ReflectionPad2d(3),
            # nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=0),
            DeformableConv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), padding=3),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(inplace_flag),
        )
            # nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
        self.encoder2 = nn.Sequential(
            DeformableConv2d(in_channels=64, out_channels=128, kernel_size=(4, 4), stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(inplace_flag),
        )

        self.encoder3 = nn.Sequential(
            # nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            DeformableConv2d(in_channels=128, out_channels=256, kernel_size=(4, 4), stride=2, padding=1),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(inplace_flag)
        )

        blocks = []
        for _ in range(residual_blocks):
            block = DResnetBlock(256, 2)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(inplace_flag),
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(inplace_flag),
        )
        self.decoder1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding=0),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        m_x = self.middle(x3)
        m_x = self.decoder3(m_x+x3)
        m_x = self.decoder2(m_x + x2)
        m_x = self.decoder1(m_x + x1)
        m_x = (torch.tanh(m_x) + 1) / 2

        return m_x


class UnetLeakyRes(BaseNetwork):
    def __init__(self, residual_blocks=8, init_weights=True):
        super(UnetLeakyRes, self).__init__()

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=0),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(inplace=True)
        )

        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(256, 2)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.LeakyReLU(inplace=True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding=0),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        h = self.encoder(x)
        h = self.middle(h)
        h = self.decoder(h)
        x = (torch.tanh(h) + 1) / 2

        return x
