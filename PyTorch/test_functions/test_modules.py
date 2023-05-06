import torch
from PyTorch.Models.DeconvModels import Deconv2D, Deconv2DMultiFilter
from PyTorch.Models.LossModules import DCT
from PyTorch.Models.ResNet import ResNet


def test_deconv_initialisation():
    pass


def test_deconv_multi_filter_dim():
    print('Testing multi-filter dims... ', end='')

    filters = 128
    deconv = Deconv2DMultiFilter(in_channels=10, out_channels=filters, kernel_size=(4, 4))
    x = torch.rand((8, 10, 18, 18))

    y = deconv(x)
    expected_size = (8, filters, 18, 18)
    assert tuple(y.size()) == expected_size

    print('PASSED')


def test_get_dct_single_img():
    expected_output = torch.tensor([[[189.3750, 11.2625, 14.2389, -11.7633, 12.1250, 39.3032, 0.7317, -11.4882],
                                     [13.0005, -9.0938, -2.5408, -34.7074, -7.2808, -0.7041, 18.5921, 27.2826],
                                     [-5.1750, 25.9718, -3.8029, -0.9106, 14.8950, -3.1129, -8.3295, 15.2390],
                                     [7.4729, -8.1367, -9.8750, 15.0937, -6.6267, 17.6210, 4.7798, -4.7374],
                                     [12.3750, 23.6339, 30.5381, 2.1903, -12.3750, 3.7889, 4.8043, -10.2701],
                                     [20.5578, -16.2111, -7.0980, -3.7616, 19.3353, 18.7785, -5.6627, 4.5097],
                                     [-3.6743, -3.2712, 9.9205, -19.1920, 8.8485, 6.8259, -10.6971, -21.4293],
                                     [32.7236, -2.6000, 9.6278, 14.0576, -8.3389, 3.7411, 0.0111, 3.7216]]])

    x = torch.tensor([45, 18, 47, 41, 14, 11, 37, 32,
                      13, 11, 43, 12, 26, 8, 10, 15,
                      20, 19, 31, 39, 17, 12, 34, 47,
                      27, 15, 28, 33, 5, 17, 27, 35,
                      45, 34, 26, 19, 1, 49, 39, 21,
                      13, 7, 1, 46, 4, 21, 22, 17,
                      40, 8, 12, 41, 40, 28, 38, 13,
                      47, 43, 5, 26, 1, 2, 6, 11.0]).view(8, 8)

    return x, expected_output


def test_dct_single_img():
    print('Testing DCT with single image... ', end='')
    x, expected_output = test_get_dct_single_img()

    x = x.view(1, 1, 8, 8)
    expected_output = expected_output.view(1, 1, 8, 8)

    dct = DCT(img_size=(8, 8))
    actual_output = dct(x)

    assert torch.allclose(actual_output, expected_output, rtol=1e-3, atol=1e-4)
    print('PASSED')


def test_dct_multi_img():
    print('Testing DCT with multiple images... ', end='')
    dct = DCT(img_size=(8, 8))

    x, y = test_get_dct_single_img()
    x = x.view(1, 1, 8, 8)
    x_in = torch.cat((x, x, x))

    y = y.view(1, 1, 8, 8)
    expected_output = torch.cat((y, y, y))

    actual_output = dct(x_in)

    assert torch.allclose(actual_output, expected_output, rtol=1e-2, atol=1e-3)
    print('PASSED')


def test_dct_multi_img_dims():
    dct = DCT(img_size=(8, 8))

    shape = (16, 3, 8, 8)

    x = torch.rand(shape)
    y = dct(x)

    assert x.size() == y.size()


def test_resnet():
    print('Testing resnet Module... ', end='')
    resnet_1 = ResNet(2, 8, channels=1)
    resnet_3 = ResNet(2, 8, channels=3)

    x_1 = torch.rand((2, 1, 16, 16))
    x_3 = torch.rand((2, 3, 16, 16))

    expected_1 = [2, 1, 32, 32]
    expected_3 = [2, 3, 32, 32]

    y_1 = resnet_1(x_1)
    y_3 = resnet_3(x_3)

    assert list(y_1.size()) == expected_1
    assert list(y_3.size()) == expected_3

    print('PASSED')


def run_tests():
    test_deconv_initialisation()
    test_deconv_multi_filter_dim()
    test_dct_single_img()
    test_dct_multi_img()
    test_resnet()
    return


if __name__ == '__main__':
    run_tests()
