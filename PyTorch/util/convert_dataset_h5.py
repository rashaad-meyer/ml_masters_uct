import os
import h5py
from matplotlib.scale import scale_factory
from requests import patch
import torch
import numpy as np
from torchvision import io
from torchvision import transforms as T
from tqdm import tqdm

from PyTorch.util.data_augmentation import RgbToYCbCr, RgbToGrayscale


def convert_div2k_to_h5(path, scale, color, output_filename, patch_size=32):
    img_names = os.listdir(path)
    img_paths = list(map(lambda img_name: f'{path}/{img_name}', img_names))
    transforms = []

    if color == 'ycbcr':
        
        transforms += [RgbToYCbCr(return_only_y=True)]
    elif color == 'grayscale':
        transforms += [RgbToGrayscale()]
    elif color != 'rgb':
        raise ValueError('Please choose a valid color space ie rgb, ycbcr, grayscale')

    hr_patches_list = []
    lr_patches_list = []

    output_dir = '/'.join(output_filename.split('/')[:-1])
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print('Processing HR into LR and ')
    for step, img_path in tqdm(enumerate(img_paths)):
        hr_img = io.read_image(img_path)

        resize = T.Resize((hr_img.shape[-2] // scale, hr_img.shape[-1] // scale),
                          interpolation=T.InterpolationMode.BICUBIC)

        for transform in transforms:
            hr_img = transform(hr_img)[0]

        lr_img = resize(hr_img)

        # Extract patches from HR and LR image
        hr_img_patches = extract_patches(hr_img, patch_size)
        lr_img_patches = extract_patches(lr_img, patch_size // scale)

        hr_patches_list.append(hr_img_patches)
        lr_patches_list.append(lr_img_patches)

        if not step < 100:
            break
    
    print('Concatenating all images patches into one tensor...')
    hr_patches = torch.cat(hr_patches_list, dim=0)
    lr_patches = torch.cat(lr_patches_list, dim=0)

    # Save to h5 file
    print(f'Saving to H5 file')
    output_filename = f'{output_filename}_x{scale}_{color}_{patch_size}.h5'
    with h5py.File(output_filename, 'w') as f:
        f.create_dataset("HR", data=hr_patches.numpy(), compression='gzip')
        f.create_dataset("LR", data=lr_patches.numpy(), compression='gzip')

    print(f'File saved to : {output_filename}...')


def extract_patches(img_tensor, patch_size=32, stride=14):
    """
    Extract patches of size patch_size x patch_size from the input tensor with a given stride.

    Parameters:
    - img_tensor: a PyTorch tensor with shape (C, H, W) where:
        - C is the number of channels (e.g., 3 for RGB images)
        - H is the height of the image
        - W is the width of the image
    - patch_size: the size of the patches to be extracted (default: 32)
    - stride: the number of pixels to skip between patches in both dimensions (default: 14)

    Returns:
    - A tensor of patches with shape (num_patches, C, patch_size, patch_size)
    """

    # Unfolding the image tensor into patches
    patches = img_tensor.unfold(1, patch_size, stride).unfold(2, patch_size, stride)

    # Reshaping the patches tensor to the desired output shape
    patches = patches.contiguous().view(-1, img_tensor.size(0), patch_size, patch_size)

    return patches


def main():
    # Example:
    # Let's say you have an RGB image tensor with shape (3, 224, 224)
    img = torch.randn(3, 224, 224)
    patches = extract_patches(img)

    # Expected to print something like (n, 3, 32, 32) depending on the number of patches extracted.
    print(patches.shape)


if __name__ == '__main__':
    convert_div2k_to_h5('data/DIV2K_train_HR', scale=2, color='ycbcr', output_filename='data/h5_datasets/div2k')
