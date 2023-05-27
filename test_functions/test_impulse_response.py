import os
import shutil
from PyTorch.Models.CnnModules import TwoLayerCNN
from PyTorch.util.impulse_response import impulse_response_of_model, save_tensor_images


def test_impulse_response():
    img_size = (32, 32)

    model = TwoLayerCNN(
        img_size=img_size,
        layer_1='deconv',
        layer_2='conv',
        layer_1_in=3,
        layer_1_out=8,
        layer_2_out=16,
        deconv_bias=False,
    )

    out = impulse_response_of_model(model, img_size)

    test_folder = 'data/impulse_response_test'
    save_tensor_images(out, 'test', test_folder)
    shutil.rmtree(test_folder)
