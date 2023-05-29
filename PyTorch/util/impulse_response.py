import os
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.fft import fft2

from PyTorch.Models.CnnModules import TwoLayerCNN


def impulse_response_of_model(model, img_size):
    model.eval()

    x = F.pad(torch.tensor([[[[1.0]]]]), (0, img_size[0] - 1, 0, img_size[1] - 1))
    _ = model(x)

    yt = model.layer1_out

    yf = fft2(yt).real

    return yf


def save_tensor_images(tensor, file_prefix, folder):
    # Get the dimensions of the tensor
    batch_size, num_channels, height, width = tensor.size()

    # if folder doesn't exist create it
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Iterate over the tensor and save each channel as a grayscale image
    for channel_idx in range(num_channels):
        # Extract the channel tensor
        channel_tensor = tensor[0, channel_idx]

        # Convert the channel tensor to a PIL image
        pil_image = TF.to_pil_image(channel_tensor)

        # Save the image with a unique filename
        filename = f"{folder}/{file_prefix}_channel_{channel_idx:02d}.png"
        pil_image.save(filename)

