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

        gm2f = torch.flip(gm1f, (0,))
        gm2f = torch.roll(gm2f, shifts=1, dims=0)

        gm3f = torch.flip(gm1f, (1,))
        gm3f = torch.roll(gm3f, shifts=1, dims=1)

        gm4f = torch.flip(gm1f, (0, 1))
        gm4f = torch.roll(gm4f, shifts=(1, 1), dims=(0, 1))

        gmf = gm1f * gm2f * gm3f * gm4f

        ymf = gmf * fft2(x)

        y = ifft2(ymf).real

        return y
