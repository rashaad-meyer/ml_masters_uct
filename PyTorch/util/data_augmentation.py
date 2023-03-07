import torch


class RandomCropIsr(object):
    def __init__(self, hr_crop_size):
        self.hr_crop_size = hr_crop_size
        pass

    def __call__(self, lr_img, hr_img):
        lr_shape = lr_img.shape[-2:]
        scale = hr_img.shape[-1] // lr_shape[-1]

        lr_crop_size = self.hr_crop_size // scale

        lr_top = torch.randint(low=0, high=lr_shape[-2] - lr_crop_size + 1, size=(1,))
        lr_left = torch.randint(low=0, high=lr_shape[-2] - lr_crop_size + 1, size=(1,))

        hr_top = lr_top * scale
        hr_left = lr_left * scale

        lr_crop = lr_img[:, lr_top:lr_top + lr_crop_size, lr_left:lr_left + lr_crop_size]
        hr_crop = hr_img[:, hr_top:hr_top + self.hr_crop_size, hr_left:hr_left + self.hr_crop_size]

        return lr_crop, hr_crop


def test():
    random_crop_isr = RandomCropIsr(32)
    hr = torch.randn((3, 64, 64))
    lr = torch.randn((3, 32, 32))
    lr_crop, hr_crop = random_crop_isr(lr, hr)

    print('LR function output shape', lr_crop.shape)
    print('LR actual output shape', (3, 16, 16))
    print('HR function output shape', hr_crop.shape)
    print('HR actual output shape', (3, 32, 32))


if __name__ == '__main__':
    test()
