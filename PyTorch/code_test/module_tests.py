import torch
import torch.nn as nn
from PyTorch.Models.DeconvModels import Deconv2D, Deconv2DMultiFilter
from PyTorch.Datasets.Datasets import ImageSuperResDataset


def deconv_initialisation_test():
    pass


def deconv_multi_filter_dim_test():
    filters = 128
    deconv = Deconv2DMultiFilter(in_channels=10, out_channels=filters, kernel_size=(4, 4))
    x = torch.rand((8, 10, 18, 18))
    y = deconv(x)
    expected_size = (8, filters, 18, 18)
    print('Actual size', tuple(y.size()))
    print('Expected size', expected_size)


def run_tests():
    deconv_initialisation_test()
    deconv_multi_filter_dim_test()


if __name__ == '__main__':
    run_tests()
