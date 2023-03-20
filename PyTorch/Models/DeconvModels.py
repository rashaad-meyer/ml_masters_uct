import torch
import torch.nn as nn
from torch.fft import fft2, ifft2


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
    def __init__(self, shape=(2, 4), filters=1):
        super(Deconv2DMultiFilter, self).__init__()

        temp = torch.rand((filters,) + shape)
        temp_sum = torch.sum(temp, dim=(1, 2))

        temp = temp / temp_sum.view((filters, 1, 1))

        temp[:, 0, 0] = 1.0

        self.w = nn.Parameter(data=temp, requires_grad=True)

    def forward(self, x):

        w = self.w
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
