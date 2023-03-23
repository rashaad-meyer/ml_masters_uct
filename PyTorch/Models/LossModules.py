import torch
import torch.nn as nn
import numpy as np


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

        self.u = (np.pi * (2 * grid_x + 1) * grid_p) / (2 * M)
        self.v = (np.pi * (2 * grid_y + 1) * grid_q) / (2 * N)

        a_p = torch.ones((P, Q)) * (2 / M) ** 0.5
        a_q = torch.ones((P, Q)) * (2 / M) ** 0.5

        a_q[0, :] = a_q[0, :] / 2.0 ** 0.5
        a_p[:, 0] = a_p[:, 0] / 2.0 ** 0.5

        self.a_q = a_q
        self.a_p = a_p

    def forward(self, img):
        img = img.unsqueeze(1).unsqueeze(1)
        y = self.a_p * self.a_q * (img * self.u.cos() * self.v.cos()).sum(dim=(-1, -2))
        return y
