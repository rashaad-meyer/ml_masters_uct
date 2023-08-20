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


class BicubicInterpolation(nn.Module):
    def __init__(self):
        super(BicubicInterpolation, self).__init__()

    def forward(self, x):
        resize = T.Resize((int(x.size(-2) * 2), int(x.size(-1) * 2)),
                          interpolation=T.InterpolationMode.BICUBIC)
        x = resize(x)
        return x
