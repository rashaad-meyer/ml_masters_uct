from typing import Any
import torch
import torchvision.transforms
from torch.nn.functional import pad


class RandomCropIsr(object):
    def __init__(self, hr_crop_size, train=True):
        self.hr_crop_size = hr_crop_size
        self.train = train

    def __call__(self, lr_img, hr_img):
        lr_shape = lr_img.shape[-2:]
        scale = hr_img.shape[-1] // lr_shape[-1]

        lr_crop_size = self.hr_crop_size // scale
        if self.train:
            lr_top = torch.randint(low=0, high=lr_shape[0] - lr_crop_size + 1, size=(1,))
            lr_left = torch.randint(low=0, high=lr_shape[1] - lr_crop_size + 1, size=(1,))
        else:
            lr_top = 50  # TODO change it to middle of image
            lr_left = 50

        hr_top = lr_top * scale
        hr_left = lr_left * scale

        lr_crop = lr_img[:, lr_top:lr_top + lr_crop_size, lr_left:lr_left + lr_crop_size]
        hr_crop = hr_img[:, hr_top:hr_top + self.hr_crop_size, hr_left:hr_left + self.hr_crop_size]

        return lr_crop, hr_crop


class PadIsr(object):
    def __init__(self, padding):
        self.padding = padding

    def __call__(self, lr_img, hr_img):
        lr_shape = lr_img.shape[-2:]
        hr_shape = hr_img.shape[-2:]
        scale = hr_shape[-1] // lr_shape[-1]

        lr_padding_size = self.padding
        hr_padding_size = self.padding * scale

        lr_pad = pad(lr_img, (lr_padding_size, lr_padding_size, lr_padding_size, lr_padding_size))
        hr_pad = pad(hr_img, (hr_padding_size, hr_padding_size, hr_padding_size, hr_padding_size))

        return lr_pad, hr_pad


class UnpadIsr(object):
    def __init__(self, pad_isr):
        self.pad_isr = pad_isr

    def __call__(self, hr_img):
        hr_padding_size = self.pad_isr.padding * 2  # 2 is because padding is added on both sides of the image

        # numpy slicing to remove the padding from all sides
        unpadded_hr_img = hr_img[..., hr_padding_size:-hr_padding_size, hr_padding_size:-hr_padding_size]

        return unpadded_hr_img


class RgbToYCbCr(object):
    def __init__(self, return_only_y=False):
        self.return_only_y = return_only_y

    def convert_image(self, image, *args: Any, **kwds: Any) -> Any:
        """Convert an RGB image to YCbCr.

        Args:
            image (torch.Tensor): RGB Image to be converted to YCbCr.

        Returns:
            torch.Tensor: YCbCr version of the image.
        """

        if not torch.is_tensor(image):
            raise TypeError("Input type is not a torch.Tensor. Got {}".format(
                type(image)))

        if len(image.shape) < 3 or image.shape[-3] != 3:
            raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                             .format(image.shape))

        r: torch.Tensor = image[..., 0, :, :]
        g: torch.Tensor = image[..., 1, :, :]
        b: torch.Tensor = image[..., 2, :, :]

        delta = .5
        y: torch.Tensor = .299 * r + .587 * g + .114 * b

        if self.return_only_y:
            return y.unsqueeze(0)

        cb: torch.Tensor = (b - y) * .564 + delta
        cr: torch.Tensor = (r - y) * .713 + delta
        return torch.stack((y, cb, cr), -3)

    def __call__(self, *images):
        return tuple(self.convert_image(img) for img in images)


class RgbToGrayscale(object):
    def __init__(self):
        self.convert_image = torchvision.transforms.Grayscale()

    def __call__(self, *images):
        return tuple(self.convert_image(img) for img in images)


def test_pad():
    pad_isr = PadIsr(32)
    hr = torch.randn((3, 64, 64))
    lr = torch.randn((3, 32, 32))
    lr_pad, hr_pad = pad_isr(lr, hr)

    print('LR function output shape', lr_pad.shape)
    print('LR expected output shape', (3, 32 + 32 * 2, 32 + 32 * 2))
    print('HR function output shape', hr_pad.shape)
    print('HR expected output shape', (3, 64 + 32 * 2 * 2, 64 + 32 * 2 * 2))


def test_random_crop():
    random_crop_isr = RandomCropIsr(32)
    hr = torch.randn((3, 64, 64))
    lr = torch.randn((3, 32, 32))
    lr_crop, hr_crop = random_crop_isr(lr, hr)

    print('LR function output shape', lr_crop.shape)
    print('LR actual output shape', (3, 16, 16))
    print('HR function output shape', hr_crop.shape)
    print('HR actual output shape', (3, 32, 32))


def test_rgb_to_ycbcr():
    rgb_to_ycbcr = RgbToYCbCr()
    image = torch.randn((3, 64, 64))
    rgb_to_ycbcr(image)
    print(image.size())


if __name__ == '__main__':
    test_rgb_to_ycbcr()
