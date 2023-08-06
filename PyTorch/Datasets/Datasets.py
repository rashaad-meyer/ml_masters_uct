import h5py
import torch
import numpy as np
from torchvision import io
from torch.utils.data import Dataset
from torchvision import transforms as T

import os


class Div2k(Dataset):
    def __init__(self, lr_path, hr_path, resize_lr=False, transform=None, ds_length=None):
        """
        Class for Image Super Resolution Dataset. It takes path for high resolution images
        and path for low resolution images. You can also transform data, specify amount of images
        you want in the dataset, and add padding if necessary.
        :param hr_path: path to folder containing high resolution images
        :param lr_path: path to folder containing low resolution images
        :param color: True for dataset to be RGB images. False for dataset to be Grayscale images
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

        hr_img = io.read_image(hr_path)
        lr_img = io.read_image(lr_path)

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
    def __init__(self, ds_path, transform=None):
        """
        Class for Image Super Resolution Dataset. It takes path for high resolution images
        and path for low resolution images. You can also transform data, specify amount of images
        you want in the dataset, and add padding if necessary.
        :param color: True for dataset to be RGB images. False for dataset to be Grayscale images
        :param transform: transformation object that takes in LR image and HR image respectively
        """
        if transform is None:
            transform = []
        self.transform = transform
        self.resize = None

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

        hr_img = io.read_image(hr_path)

        self.resize = T.Resize((hr_img.size()[-2] // 2, hr_img.size()[-1] // 2),
                               interpolation=T.InterpolationMode.BICUBIC)
        lr_img = self.resize(hr_img)

        for transform in self.transform:
            lr_img, hr_img = transform(lr_img, hr_img)

        # Preprocess HR image
        hr_img = hr_img / 255.0

        # Preprocess LR image
        lr_img = lr_img / 255.0

        return lr_img, hr_img


class TrainDataset(Dataset):
    def __init__(self, h5_file, same_size=True):
        super(TrainDataset, self).__init__()
        self.h5_file = h5_file
        self.scale = int(h5_file[-4])
        self.same_size = same_size

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            hr_img = torch.from_numpy(np.expand_dims(f['hr'][idx] / 255., 0))

            if self.same_size:
                lr_img = torch.from_numpy(np.expand_dims(f['lr'][idx] / 255., 0))
            else:
                if self.scale % 2 == 0:
                    hr_img = hr_img[..., :32, :32]

                resize = T.Resize((hr_img.size()[-2] // self.scale, hr_img.size()[-1] // self.scale),
                                  interpolation=T.InterpolationMode.BICUBIC)

                lr_img = resize(hr_img)
            return lr_img, hr_img

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])


class EvalDataset(Dataset):
    def __init__(self, h5_file, same_size=True):
        super(EvalDataset, self).__init__()
        self.h5_file = h5_file
        self.scale = int(h5_file[-4])
        self.same_size = same_size

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            hr_img = torch.from_numpy(np.expand_dims(f['hr'][str(idx)][:, :] / 255., 0))

            if self.same_size:
                lr_img = torch.from_numpy(np.expand_dims(f['lr'][str(idx)][:, :] / 255., 0))
            else:
                if self.scale % 2 == 0:
                    crop_size_h = hr_img.size()[-2] - (hr_img.size()[-2] % self.scale)
                    crop_size_w = hr_img.size()[-1] - (hr_img.size()[-1] % self.scale)
                    hr_img = hr_img[..., :crop_size_h, :crop_size_w]

                resize = T.Resize((hr_img.size()[-2] // self.scale, hr_img.size()[-1] // self.scale),
                                  interpolation=T.InterpolationMode.BICUBIC)

                lr_img = resize(hr_img)
            return lr_img, hr_img

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])


if __name__ == '__main__':
    # print(os.listdir(''))
    ds = TrainDataset('../../data/sr/srcnn/91-image_x2.h5', same_size=False)
    val_ds = EvalDataset('../../data/sr/srcnn/Set5_x2.h5', same_size=False)

    for i in range(10):
        HR_img, LR_img = ds[i]
        print(HR_img.shape, LR_img.shape)

    for i in range(5):
        HR_img, LR_img = val_ds[i]
        print(HR_img.shape, LR_img.shape)
