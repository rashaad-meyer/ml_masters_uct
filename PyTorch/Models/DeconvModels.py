import torch
import torch.nn as nn
from torch.fft import fft2, ifft2


def __init__():
    pass


class Deconv2D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, kernel_size=(2, 4), bias=True, first_elem_trainable=False,
                 four_factor=True):
        super(Deconv2D, self).__init__()

        init_factor = (kernel_size[0] * kernel_size[1] * (in_channels + out_channels))

        if first_elem_trainable:
            # initialise filter as correct shape so that first element is trainable
            w = torch.randn((out_channels, in_channels,) + kernel_size)
            w = w / init_factor
            w[:, :, 0, 0] = 1.0

        else:
            # initialise filter as flat to be reshaped after so that first element is not trainable
            w = torch.randn((out_channels, in_channels,) + (kernel_size[0] * kernel_size[1] - 1,))
            w = w / init_factor

        w = nn.Parameter(data=w, requires_grad=True)

        # make first element of each filter 1.0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.w = w
        self.first_elem_trainable = first_elem_trainable
        self.four_factor = four_factor
        self.conv = nn.Conv2d(3, 1, 3, padding='same')

        if bias:
            self.b = nn.Parameter(data=torch.rand(1, out_channels, 1, 1) - 0.5, requires_grad=True)
        else:
            self.b = nn.Parameter(data=torch.tensor([0.0]), requires_grad=False)

    def forward(self, x):
        # add dimension so that we can broadcast it later
        x = x.unsqueeze(dim=1)

        w = self.w
        if not self.first_elem_trainable:
            w = nn.functional.pad(w, (1, 0), value=1)
            w = torch.reshape(w, (self.out_channels, self.in_channels,) + self.kernel_size)

        hm1 = nn.functional.pad(w, (0, x.size(-1) - w.size(-1), 0, x.size(-2) - w.size(-2)))

        gm1f = 1 / fft2(hm1)

        if self.four_factor:
            gm2f_ = torch.flip(gm1f, dims=(-2,))
            gm2f = torch.roll(gm2f_, shifts=1, dims=-2)

            gm3f_ = torch.flip(gm1f, dims=(-1,))
            gm3f = torch.roll(gm3f_, shifts=1, dims=-1)

            gm4f_ = torch.flip(gm1f, dims=(-2, -1))
            gm4f = torch.roll(gm4f_, shifts=(1, 1), dims=(-2, -1))

            gmf = gm1f * gm2f * gm3f * gm4f
        else:
            gmf = gm1f

        ymf = gmf * fft2(x)

        y = ifft2(ymf).real

        # sum along input channels dim to reduce dims to standard image dims (batch x channels x height x width)
        batch, in_channels, out_channels, img_height, img_width = y.size()
        y = self.conv(y.view(-1, in_channels, img_height, img_width))
        y = y.view(batch, out_channels, img_height, img_width)

        return y
