from PyTorch.util.data_augmentation import RgbToYCbCr, RgbToGrayscale

import os
import h5py
from tqdm import tqdm
import torch
from torchvision import io, transforms as T


# ... Your previous functions and imports ...

def convert_div2k_to_h5(path, scale, color, output_filename, patch_size=32, stride=32):
    img_names = os.listdir(path)
    img_paths = list(map(lambda img_name: f'{path}/{img_name}', img_names))
    transforms = []

    num_channels = 3
    if color == 'ycbcr':
        transforms += [RgbToYCbCr(return_only_y=True)]
        num_channels = 1
    elif color == 'grayscale':
        transforms += [RgbToGrayscale()]
        num_channels = 1
    elif color != 'rgb':
        raise ValueError('Please choose a valid color space ie rgb, ycbcr, grayscale')

    output_dir = os.path.dirname(output_filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_filename = f'{output_filename}_x{scale}_{color}_{patch_size}.h5'

    if os.path.exists(output_filename):
        print(f'{output_filename.split("/")[-1]} already exists in {output_dir}')
        return output_filename

    print('Computing max size of h5 dataset')
    max_size = calculate_total_patches(img_paths, patch_size, stride)
    # Open the HDF5 file
    with h5py.File(output_filename, 'w') as f:

        # Create empty datasets for HR and LR with maximum possible size
        hr_dset = f.create_dataset("HR",
                                   (max_size, num_channels, patch_size, patch_size),
                                   dtype='float32',
                                   compression='gzip',
                                   maxshape=(None, num_channels, patch_size, patch_size))
        lr_dset = f.create_dataset("LR",
                                   (max_size, num_channels, patch_size // scale, patch_size // scale),
                                   dtype='float32',
                                   compression='gzip',
                                   maxshape=(None, num_channels, patch_size // scale, patch_size // scale))

        count_hr = 0
        count_lr = 0

        for counter, img_path in enumerate(img_paths):
            print(f'Processing image {counter + 1:03d}/{len(img_paths)}')

            hr_img = io.read_image(img_path)

            resize = T.Resize((hr_img.shape[-2] // scale, hr_img.shape[-1] // scale),
                              interpolation=T.InterpolationMode.BICUBIC)

            for transform in transforms:
                hr_img = transform(hr_img)[0]

            lr_img = resize(hr_img)

            # Extract patches from HR and LR image
            hr_img_patches = extract_patches(hr_img, patch_size, stride=stride)
            lr_img_patches = extract_patches(lr_img, patch_size // scale, stride=stride // scale)

            # Try to write patches directly to the HDF5 datasets
            try:
                hr_dset[count_hr: count_hr + hr_img_patches.size(0)] = hr_img_patches.numpy()
                lr_dset[count_lr: count_lr + lr_img_patches.size(0)] = lr_img_patches.numpy()
            except Exception as e:
                print(e)
                break

            count_hr += hr_img_patches.size(0)
            count_lr += lr_img_patches.size(0)

        # Resize datasets to fit the actual size
        hr_dset.resize(count_hr, axis=0)
        lr_dset.resize(count_lr, axis=0)


def calculate_total_patches(img_paths, patch_size, stride):
    total_patches = 0
    for img_path in tqdm(img_paths):
        hr_img = io.read_image(img_path)
        h, w = hr_img.shape[-2], hr_img.shape[-1]
        num_patches_h = (h - patch_size) // stride + 1
        num_patches_w = (w - patch_size) // stride + 1
        total_patches += num_patches_h * num_patches_w
    return total_patches


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

    return patches / 255.0


def main():
    # Example:
    # Let's say you have an RGB image tensor with shape (3, 224, 224)
    img = torch.randn(3, 224, 224)
    patches = extract_patches(img)

    # Expected to print something like (n, 3, 32, 32) depending on the number of patches extracted.
    print(patches.shape)


if __name__ == '__main__':
    convert_div2k_to_h5('../../data/DIV2K_train_HR', scale=2, color='ycbcr', output_filename='../../data/div2k_patches')
