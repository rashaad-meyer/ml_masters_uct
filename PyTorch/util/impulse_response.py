import os
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.fft import fft2

from PyTorch.Models.CnnModules import TwoLayerCNN


def impulse_response_of_model(model, img_size):
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    x = F.pad(torch.tensor([[[[1.0]]]]), (0, img_size[-2] - 1, 0, img_size[-1] - 1))
    x = x.to(device)

    _ = model(x)
    y = fft2(model.layer1_out)

    # return output magnitude and phase
    return y.abs(), y.angle()


def save_tensor_images(tensor, file_prefix=None, folder=None):
    # Get the dimensions of the tensor
    batch_size, num_channels, height, width = tensor.size()

    # if folder doesn't exist create it
    if file_prefix is not None and folder is not None:
        if not os.path.exists(folder):
            os.makedirs(folder)

    pil_images = []

    transform = T.ToPILImage()
    # Iterate over the tensor and save each channel as a grayscale image
    for channel_idx in range(batch_size):
        # Extract the channel tensor
        channel_tensor = tensor[0, channel_idx]

        # Convert the channel tensor to a PIL image
        pil_image = transform(channel_tensor.unsqueeze(0))

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