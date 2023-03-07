from PyTorch.Datasets.Datasets import ImageSuperResDataset
from PyTorch.util.data_augmentation import RandomCropIsr
from torch.utils.data import DataLoader
import os


def test_isr_data():
    hr_path = '../../../../data/deconv_tests/DIV2K/DIV2K_train_HR'
    lr_path = '../../../../data/deconv_tests/DIV2K/DIV2K_train_LR_bicubic/X2'

    random_crop = RandomCropIsr(96)

    data = ImageSuperResDataset(lr_path, hr_path, transform=random_crop, ds_length=80)
    dataloader = DataLoader(data, batch_size=16, shuffle=True)

    x_batch, y_batch = next(iter(dataloader))

    print(x_batch.size(), y_batch.size())
    print(len(data))


if __name__ == '__main__':
    for i in range(5):
        test_isr_data()
