import torch


def random_crop_isr(lr_img, hr_img, hr_crop_size):
    lr_shape = lr_img.shape[-2:]
    scale = hr_img.shape[-1] // lr_shape[-1]

    lr_crop_size = hr_crop_size // scale

    lr_top = torch.randint(low=0, high=lr_shape[-2] - lr_crop_size + 1, size=(1,))
    lr_left = torch.randint(low=0, high=lr_shape[-2] - lr_crop_size + 1, size=(1,))

    hr_top = lr_top * scale
    hr_left = lr_left * scale

    lr_crop = lr_img[:, lr_top:lr_top + lr_crop_size, lr_left:lr_left + lr_crop_size]
    hr_crop = hr_img[:, hr_top:hr_top + hr_crop_size, hr_left:hr_left + hr_crop_size]

    return lr_crop, hr_crop


def test():
    hr = torch.randn((3, 64, 64))
    lr = torch.randn((3, 32, 32))
    lr_crop, hr_crop = random_crop_isr(lr, hr, 32)

    print('LR function output shape', lr_crop.shape)
    print('LR actual output shape', (3, 16, 16))
    print('HR function output shape', hr_crop.shape)
    print('HR actual output shape', (3, 32, 32))


if __name__ == '__main__':
    test()
