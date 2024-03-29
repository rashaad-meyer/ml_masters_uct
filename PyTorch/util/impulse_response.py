import os
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.fft import fft2, fftshift

from PyTorch.Models.CnnModules import TwoLayerCNN


def impulse_response_of_model(model, img_size):
    """

    :param model:
    :param img_size: (C x H x W)
    :return:
    """
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    x = torch.zeros(img_size)

    x[0, x.size(-2) // 2, x.size(-1) // 2] = 1.0
    x[1, x.size(-2) // 2, x.size(-1) // 2] = 1.0
    x[2, x.size(-2) // 2, x.size(-1) // 2] = 1.0

    num_channels = img_size[0]
    x = x.expand(num_channels, -1, -1).unsqueeze(0)

    x = x.to(device)

    with torch.no_grad():
        _ = model(x)
        y = model.layer1_out
        yf = fft2(y)
        yf = fftshift(yf, dim=(-1, -2))

    # return output magnitude and phase
    return yf.abs(), yf.angle(), y


def image_response_of_model(model, image):
    model.eval()

    with torch.no_grad():
        _ = model(image)
        y = model.layer1_out

    return y


def save_tensor_images(tensor, file_prefix=None, folder=None):
    # Get the dimensions of the tensor
    tensor = tensor.squeeze()

    if len(tensor.size()) < 3:
        tensor = tensor.unsqueeze(0)

    batch_size = tensor.size(0)
    # if folder doesn't exist create it
    if file_prefix is not None and folder is not None:
        if not os.path.exists(folder):
            os.makedirs(folder)

    pil_images = []

    transform = T.ToPILImage()
    # Iterate over the tensor and save each channel as a grayscale image
    for channel_idx in range(batch_size):
        # Extract the channel tensor
        channel_tensor = tensor[channel_idx]

        # Convert the channel tensor to a PIL image
        if len(channel_tensor.size()) == 2:
            pil_image = transform(channel_tensor.unsqueeze(0))
        else:
            pil_image = transform(channel_tensor)

        pil_images.append(pil_image)

        # Save the image with a unique filename
        if file_prefix is not None and folder is not None:
            filename = f"{folder}/{file_prefix}_channel_{channel_idx:02d}.png"
            pil_image.save(filename)

    return pil_images


def check_filter_diff(tensor):
    batch_size, num_channels, height, width = tensor.size()
    diff_all = []
    for i in range(num_channels):
        # Extract the channel tensor
        channel_tensor_01 = tensor[0, i]
        diffs = []

        for j in range(num_channels):
            channel_tensor_02 = tensor[0, j]
            diff = (channel_tensor_01 - channel_tensor_02).abs().sum().item()
            diffs.append(diff)

        diff_all.append(diffs)
    return torch.tensor(diff_all).view(1, 1, num_channels, num_channels)
