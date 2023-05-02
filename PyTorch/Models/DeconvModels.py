import torch
import torch.nn as nn
from torch.fft import fft2, ifft2


def __init__():
    pass


class Deconv2D(nn.Module):
    def __init__(self, shape=(2, 4)):
        super(Deconv2D, self).__init__()
        self.w_flat = nn.Parameter(data=torch.zeros(shape[0] * shape[1] - 1),
                                   requires_grad=True)
        self.h_shape = shape

    def forward(self, x):
        w = nn.functional.pad(self.w_flat, (1, 0), value=1)

        w = torch.reshape(w, self.h_shape)
        hm1 = nn.functional.pad(w, (0, x.size(-1) - w.size(-1), 0, x.size(-2) - w.size(-2)))

        gm1f = 1 / fft2(hm1)

        gm2f_ = torch.flip(gm1f, (0,))
        gm2f = torch.roll(gm2f_, shifts=1, dims=0)

        gm3f_ = torch.flip(gm1f, (1,))
        gm3f = torch.roll(gm3f_, shifts=1, dims=1)

        gm4f_ = torch.flip(gm1f, (0, 1))
        gm4f = torch.roll(gm4f_, shifts=(1, 1), dims=(0, 1))

        gmf = gm1f * gm2f * gm3f * gm4f

        ymf = gmf * fft2(x)

        y = ifft2(ymf).real

        return y


class Deconv2DWithScaling(nn.Module):
    def __init__(self, shape):
        super(Deconv2DWithScaling, self).__init__()
        self.deconv = Deconv2D(shape)
        self.w = nn.Parameter(data=torch.tensor([0.5]), requires_grad=True)

    def forward(self, x):
        x = self.deconv(x)
        x = self.w * x
        return x


class Deconv2DMultiFilter(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, kernel_size=(2, 4), ):
        super(Deconv2DMultiFilter, self).__init__()

        # initialise filter weights
        w = torch.rand((out_channels, in_channels,) + (kernel_size[0] * kernel_size[1] - 1,))

        # divide each filter by the sum of their weights multiplied by number of input channels so that
        # out_channel filters don't increase in
        init_factor = (kernel_size[0] * kernel_size[1] * (in_channels + out_channels))
        w = w / init_factor

        w = nn.Parameter(data=w, requires_grad=True)

        # make first element of each filter 1.0
        # w[:, :, 0, 0] = 1.0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.w = w
        self.b = nn.Parameter(data=torch.rand(1, out_channels, 1, 1) - 0.5, requires_grad=True)

    def forward(self, x):
        # add dimension so that we can broadcast it later
        x = x.unsqueeze(dim=1)

        w = self.w
        w = nn.functional.pad(w, (1, 0), value=1)
        w = torch.reshape(w, (self.out_channels, self.in_channels,) + self.kernel_size)

        hm1 = nn.functional.pad(w, (0, x.size(-1) - w.size(-1), 0, x.size(-2) - w.size(-2)))

        gm1f = 1 / fft2(hm1)

        gm2f_ = torch.flip(gm1f, (0,))
        gm2f = torch.roll(gm2f_, shifts=1, dims=0)

        gm3f_ = torch.flip(gm1f, (1,))
        gm3f = torch.roll(gm3f_, shifts=1, dims=1)

        gm4f_ = torch.flip(gm1f, (0, 1))
        gm4f = torch.roll(gm4f_, shifts=(1, 1), dims=(0, 1))

        gmf = gm1f * gm2f * gm3f * gm4f

        ymf = gmf * fft2(x)

        y = ifft2(ymf).real

        # sum along input channels dim to reduce dims to standard image dims (batch x channels x height x width)
        y = torch.mean(y, dim=-3) + self.b

        return y
