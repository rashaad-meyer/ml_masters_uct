import torch
import torch.nn as nn
import numpy as np
from PyTorch.util.ssim import ssim, ms_ssim, fspecial_gauss_1d


class DCT(nn.Module):
    def __init__(self, img_size):
        super(DCT, self).__init__()
        """
        Applies Discrete Cosine Transform to image that is passed into it.
        Image have to be rank 4 with dimensions corresponding to:
        batch x channels x height x width
        The image size needs to be passed when initialising the layer so
        that tensors don't have to be calculated each time the object is called
        """

        M = img_size[-2]
        N = img_size[-1]
        m = torch.tensor([x for x in range(M)])
        n = torch.tensor([x for x in range(N)])

        grid_y, grid_x = torch.meshgrid(m, n, indexing='ij')
        P = M
        Q = N
        p = torch.tensor([x for x in range(P)])
        q = torch.tensor([x for x in range(Q)])

        grid_q, grid_p = torch.meshgrid(p, q, indexing='ij')

        grid_q = grid_q.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        grid_p = grid_p.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)

        u = (np.pi * (2 * grid_x + 1) * grid_p) / (2 * M)
        v = (np.pi * (2 * grid_y + 1) * grid_q) / (2 * N)

        a_p = torch.ones((P, Q)) * (2 / M) ** 0.5
        a_q = torch.ones((P, Q)) * (2 / M) ** 0.5

        a_q[0, :] = a_q[0, :] / 2.0 ** 0.5
        a_p[:, 0] = a_p[:, 0] / 2.0 ** 0.5

        self.a_q = nn.Parameter(a_q, requires_grad=False)
        self.a_p = nn.Parameter(a_p, requires_grad=False)
        self.u = nn.Parameter(u, requires_grad=False)
        self.v = nn.Parameter(v, requires_grad=False)

    def forward(self, img):
        img = img.unsqueeze(1).unsqueeze(1)
        y = self.a_p * self.a_q * (img * self.u.cos() * self.v.cos()).sum(dim=(-1, -2))
        return y


class MSE_WITH_DCT(nn.Module):
    def __init__(self, img_size=(96, 96)):
        super(MSE_WITH_DCT, self).__init__()
        self.mse = nn.MSELoss()
        self.dct = DCT(img_size)

    def forward(self, x, y):
        x = self.dct(x)
        y = self.dct(y)
        loss = self.mse(x, y)
        return loss


class SSIM(torch.nn.Module):
    def __init__(
            self,
            data_range=255,
            size_average=True,
            win_size=11,
            win_sigma=1.5,
            channel=3,
            spatial_dims=2,
            K=(0.01, 0.03),
            nonnegative_ssim=False,
    ):
        r""" class for ssim
        Args:
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            channel (int, optional): input channels (default: 3)
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
            nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu.
        """

        super(SSIM, self).__init__()
        self.win_size = win_size
        self.win = fspecial_gauss_1d(win_size, win_sigma).repeat([channel, 1] + [1] * spatial_dims)
        self.size_average = size_average
        self.data_range = data_range
        self.K = K
        self.nonnegative_ssim = nonnegative_ssim

    def forward(self, X, Y):
        return ssim(
            X,
            Y,
            data_range=self.data_range,
            size_average=self.size_average,
            win=self.win,
            K=self.K,
            nonnegative_ssim=self.nonnegative_ssim,
        )


class MS_SSIM(torch.nn.Module):
    def __init__(
            self,
            data_range=255,
            size_average=True,
            win_size=11,
            win_sigma=1.5,
            channel=3,
            spatial_dims=2,
            weights=None,
            K=(0.01, 0.03),
    ):
        r""" class for ms-ssim
        Args:
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            channel (int, optional): input channels (default: 3)
            weights (list, optional): weights for different levels
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        """

        super(MS_SSIM, self).__init__()
        self.win_size = win_size
        self.win = fspecial_gauss_1d(win_size, win_sigma).repeat([channel, 1] + [1] * spatial_dims)
        self.size_average = size_average
        self.data_range = data_range
        self.weights = weights
        self.K = K

    def forward(self, X, Y):
        return ms_ssim(
            X,
            Y,
            data_range=self.data_range,
            size_average=self.size_average,
            win=self.win,
            weights=self.weights,
            K=self.K,
        )
