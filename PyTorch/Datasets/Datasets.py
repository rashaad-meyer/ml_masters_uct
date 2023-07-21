from torch.utils.data import Dataset
from torchvision import io
from torchvision import transforms as T

import os


class Div2k(Dataset):
    def __init__(self, lr_path, hr_path, resize_lr=False, rgb=False, transform=None, ds_length=None):
        """
        Class for Image Super Resolution Dataset. It takes path for high resolution images
        and path for low resolution images. You can also transform data, specify amount of images
        you want in the dataset, and add padding if necessary.
        :param hr_path: path to folder containing high resolution images
        :param lr_path: path to folder containing low resolution images
        :param rgb: True for dataset to be RGB images. False for dataset to be Grayscale images
        :param transform: transformation object that takes in LR image and HR image respectively
        :param ds_length: Length that you would like the dataset to be
        """
        if transform is None:
            transform = []
        self.hr_path = hr_path
        self.lr_path = lr_path
        self.transform = transform
        self.resize_lr = resize_lr
        self.resize = None

        if rgb:
            self.color_mode = io.ImageReadMode.UNCHANGED
        else:
            self.color_mode = io.ImageReadMode.GRAY

        y_imgs = os.listdir(hr_path)
        x_imgs = os.listdir(lr_path)

        # sort lists
        y_imgs.sort()
        x_imgs.sort()

        # shorten image list
        y_imgs = y_imgs[:ds_length]
        x_imgs = x_imgs[:ds_length]

        self.hr_paths = list(map(lambda img_path: f'{hr_path}/{img_path}', y_imgs))
        self.lr_paths = list(map(lambda img_path: f'{lr_path}/{img_path}', x_imgs))

    def __len__(self):
        return len(self.hr_paths)

    def __getitem__(self, idx):
        hr_path = self.hr_paths[idx]
        lr_path = self.lr_paths[idx]

        hr_img = io.read_image(hr_path, self.color_mode)
        lr_img = io.read_image(lr_path, self.color_mode)

        for transform in self.transform:
            lr_img, hr_img = transform(lr_img, hr_img)

        if self.resize_lr:
            if self.resize is None:
                self.resize = T.Resize(hr_img.size()[-2:], interpolation=T.InterpolationMode.BICUBIC)
            lr_img = self.resize(lr_img)

        # Preprocess HR image
        hr_img = hr_img / 255.0

        # Preprocess LR image
        lr_img = lr_img / 255.0

        return lr_img, hr_img


class IsrEvalDatasets(Dataset):
    def __init__(self, ds_path, rgb=False, transform=None):
        """
        Class for Image Super Resolution Dataset. It takes path for high resolution images
        and path for low resolution images. You can also transform data, specify amount of images
        you want in the dataset, and add padding if necessary.
        :param hr_path: path to folder containing high resolution images
        :param lr_path: path to folder containing low resolution images
        :param rgb: True for dataset to be RGB images. False for dataset to be Grayscale images
        :param transform: transformation object that takes in LR image and HR image respectively
        :param ds_length: Length that you would like the dataset to be
        """
        self.transform = transform
        self.resize = None

        if rgb:
            self.color_mode = io.ImageReadMode.UNCHANGED
        else:
            self.color_mode = io.ImageReadMode.GRAY

        y_imgs = []

        # Get a list of all files in the specified folder
        ds_path = f'{ds_path}/image_SRF_2'
        all_files = os.listdir(ds_path)

        # Iterate over the files
        for filename in all_files:
            # Check if the filename ends with HR.png
            if filename.endswith("HR.png"):
                # If it does, add it to the y_imgs list
                y_imgs.append(filename)

        # sort lists
        y_imgs.sort()

        self.hr_paths = list(map(lambda img_path: f'{ds_path}/{img_path}', y_imgs))

    def __len__(self):
        return len(self.hr_paths)

    def __getitem__(self, idx):
        hr_path = self.hr_paths[idx]

        hr_img = io.read_image(hr_path, self.color_mode)

        self.resize = T.Resize((hr_img.size()[-2] // 2, hr_img.size()[-1] // 2),
                               interpolation=T.InterpolationMode.BICUBIC)
        lr_img = self.resize(hr_img)

        if self.transform:
            lr_img, hr_img = self.transform(lr_img, hr_img)

        # Preprocess HR image
        hr_img = hr_img / 255.0

        # Preprocess LR image
        lr_img = lr_img / 255.0

        return lr_img, hr_img
