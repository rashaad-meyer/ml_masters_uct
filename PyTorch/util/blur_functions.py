import torch
import torch.nn as nn
import numpy as np
from torch.fft import fft2, ifft2


def blur_images(img, w=torch.FloatTensor([[1, 0.2, 0, 0], [0, 0, 0, 0]])):
    # 4 factor blur

    hm1 = nn.functional.pad(w, (0, img.size(-1) - w.size(-1), 0, img.size(-2) - w.size(-2)))

    gm1f = fft2(hm1)

    gm2f = torch.flip(gm1f, (0,))
    gm2f = torch.roll(gm2f, shifts=1, dims=0)

    gm3f = torch.flip(gm1f, (1,))
    gm3f = torch.roll(gm3f, shifts=1, dims=1)

    gm4f = torch.flip(gm1f, (0, 1))
    gm4f = torch.roll(gm4f, shifts=(1, 1), dims=(0, 1))

    gmf = gm1f * gm2f * gm3f * gm4f

    ymf = gmf * fft2(img)

    X = ifft2(ymf).real

    return X


def gaussian_kernel(size, sigma=1):
    assert size % 2 == 1
    kernel = np.zeros((size, size))
    center = size // 2

    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

    kernel = kernel / kernel.sum()
    kernel = kernel[center:, center:]
    kernel[0, 0] = 1.0
    return kernel
