import torch
import torch.nn as nn

from PyTorch.Models.DeconvModels import Deconv2D


class ResBlock(nn.Module):
    def __init__(self, n_feat, kernel_size, res_scale=1.0, bias=True, bn=False, activation=nn.ReLU(True)):
        super(ResBlock, self).__init__()
        modules = []

        for i in range(2):
            conv = nn.Conv2d(n_feat, n_feat, kernel_size, padding='same', bias=bias)
            modules.append(conv)
            if bn:
                modules.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                modules.append(activation)

        self.block = nn.Sequential(*modules)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.block(x).mul(self.res_scale)
        res += x

        return res


class UpScale(nn.Sequential):
    def __init__(self, factor, n_feat, bias=True):
        modules = [
            nn.Conv2d(n_feat, n_feat * factor ** 2, 3, padding='same', bias=bias),
            nn.PixelShuffle(2)
        ]

        super(UpScale, self).__init__(*modules)


class ResNet(nn.Module):
    def __init__(self, n_resblocks, n_features, deconv=False, channels=1):
        super().__init__()
        if deconv:
            self.start = Deconv2D(channels, n_features, (3, 3), first_elem_trainable=True)
        else:
            self.start = nn.Conv2d(channels, n_features, (3, 3), padding='same')

        resblocks = []
        for i in range(n_resblocks):
            resblocks.append(ResBlock(n_features, (3, 3), res_scale=0.15))

        resblocks.append(nn.Conv2d(n_features, n_features, 3, padding='same', bias=True))

        upsampler = UpScale(2, n_features)

        self.resblocks = nn.Sequential(*resblocks)
        self.upsampler = upsampler
        self.end = nn.Conv2d(n_features, channels, 3, padding='same')

    def forward(self, x):
        xs = self.start(x)
        x = self.resblocks(xs)
        x = self.upsampler(x + 0.15*xs)
        x = self.end(x)
        return x

