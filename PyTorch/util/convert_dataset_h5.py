import os
import torch
from torchvision import io
from torchvision import transforms as T
from torchvision.utils import save_image
from tqdm import tqdm

from PyTorch.util.data_augmentation import RgbToYCbCr, RgbToGrayscale


def convert_div2k_to_h5(path, scale, color, output_dir, patch_size=32):
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

    if not os.path.exists(output_dir):
        os.makedirs(f'{output_dir}/hr')
        os.makedirs(f'{output_dir}/lr')
    else:
        print('Folder for patches already created. If you would like to regenerate the patche delete the folder: '
              f'{output_dir}')
        return output_dir

    print('Processing HR into LR and ')
    for step, img_path in enumerate(img_paths):
        print(f'Processing {step:04d}/{len(img_paths)}')
        hr_img = io.read_image(img_path)
        img_name = img_path.split('/')[-1]

        resize = T.Resize((hr_img.shape[-2] // scale, hr_img.shape[-1] // scale),
                          interpolation=T.InterpolationMode.BICUBIC)

        for transform in transforms:
            hr_img = transform(hr_img)[0]

        lr_img = resize(hr_img)

        # Extract patches from HR and LR image
        hr_img_patches = extract_patches(hr_img, patch_size)
        lr_img_patches = extract_patches(lr_img, patch_size // scale)

        for i, (hr_img_patch, lr_img_patch) in enumerate(zip(hr_img_patches, lr_img_patches)):
            # save patches
            save_image(hr_img_patch, f'{output_dir}/hr/{img_name[:-4]}_{i:04d}.png')
            save_image(lr_img_patch, f'{output_dir}/lr/{img_name[:-4]}_{i:04d}.png')

    return f'{output_dir}/lr', f'{output_dir}/hr'


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
    convert_div2k_to_h5('../../data/DIV2K_train_HR', scale=2, color='ycbcr', output_dir='../../data/div2k_y')
