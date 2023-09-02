import logging
import os

import torch.nn as nn
from torchvision import transforms as T
from PyTorch.Models.DeconvModels import Deconv2D


class SRCNN(nn.Module):
    def __init__(self, num_channels=1, channels_1=64, channels_2=32, upscale_factor=2, deconv=False,
                 use_pixel_shuffle=True, bias=True, first_elem_trainable=False, pad_inner=None, four_factor=True):
        super(SRCNN, self).__init__()

        if deconv:
            self.conv1 = Deconv2D(num_channels, channels_1, (9, 9), bias=bias, four_factor=four_factor,
                                  first_elem_trainable=first_elem_trainable, pad_inner=pad_inner)
        else:
            self.conv1 = nn.Conv2d(num_channels, channels_1, kernel_size=9, padding=4)

        self.conv2 = nn.Conv2d(channels_1, channels_2, kernel_size=5, padding='same')

        if use_pixel_shuffle:
            self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
            self.conv3 = nn.Conv2d(channels_2, num_channels * upscale_factor ** 2, kernel_size=5, padding=2)
        else:
            self.pixel_shuffle = nn.Identity()
            self.conv3 = nn.Conv2d(channels_2, num_channels, kernel_size=5, padding=2)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        x = self.pixel_shuffle(x)
        return x


class SrCnnPixelShuffle(nn.Module):
    def __init__(self, num_channels=1, channels_1=64, channels_2=32, upscale_factor=2, deconv=False,
                 bias=True, first_elem_trainable=False, pad_inner=None, four_factor=True):

        # Configure logging
        log_path = 'logs/output.log'
        os.makedirs('/'.join(log_path.split('/')[:-1]), exist_ok=True)
        self.logger = logging.getLogger(__name__)
        handler = logging.FileHandler(log_path)
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

        super(SrCnnPixelShuffle, self).__init__()

        if deconv:
            self.conv1 = Deconv2D(num_channels, channels_1, (5, 5), bias=bias, four_factor=four_factor,
                                  first_elem_trainable=first_elem_trainable, pad_inner=pad_inner)
        else:
            self.conv1 = nn.Conv2d(num_channels, channels_1, kernel_size=5, padding='same')

        self.conv2 = nn.Conv2d(channels_1, channels_2, kernel_size=3, padding='same')

        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.conv3 = nn.Conv2d(channels_2, num_channels * upscale_factor ** 2, kernel_size=3, padding='same')

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        self.logger.info('conv1 output: {:.3f}'.format(x.mean().item()))
        x = self.tanh(x)
        self.logger.info('tanh1 output: {:.3f}'.format(x.mean().item()))

        x = self.tanh(self.conv2(x))
        # self.logger.info('conv2 output: {:.3f}'.format(x.mean().item()))

        x = self.conv3(x)
        # self.logger.info('conv3 output: {:.3f}'.format(x.mean().item()))

        x = self.pixel_shuffle(x)
        x = self.sigmoid(x)
        # self.logger.info('PsSig output: {:.3f}'.format(x.mean().item()))
        self.logger.info('=' * 30)
        return x


class BicubicInterpolation(nn.Module):
    def __init__(self):
        super(BicubicInterpolation, self).__init__()

    def forward(self, x):
        resize = T.Resize((int(x.size(-2) * 2), int(x.size(-1) * 2)),
                          interpolation=T.InterpolationMode.BICUBIC)
        x = resize(x)
        return x
