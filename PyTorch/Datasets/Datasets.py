from torch.utils.data import Dataset
from torchvision import io

import os


class ImageSuperResDataset(Dataset):
    def __init__(self, lr_path, hr_path, rgb=False, transform=None, ds_length=None):
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
        self.hr_path = hr_path
        self.lr_path = lr_path
        self.transform = transform

        if rgb:
            self.color_mode = io.ImageReadMode.UNCHANGED
        else:
            self.color_mode = io.ImageReadMode.GRAY

        y_imgs = os.listdir(hr_path)[:ds_length]
        x_imgs = os.listdir(lr_path)[:ds_length]

        self.y_paths = list(map(lambda img_path: f'{hr_path}/{img_path}', y_imgs))
        self.x_paths = list(map(lambda img_path: f'{lr_path}/{img_path}', x_imgs))

    def __len__(self):
        return len(self.y_paths)

    def __getitem__(self, idx):
        y_path = self.y_paths[idx]
        x_path = self.x_paths[idx]

        y_img = io.read_image(y_path, self.color_mode)
        x_img = io.read_image(x_path, self.color_mode)

        if self.transform:
            x_img, y_img = self.transform(x_img, y_img)

        # Preprocess HR image
        y_img = y_img / 255.0

        # Preprocess LR image
        x_img = x_img / 255.0

        return x_img, y_img
