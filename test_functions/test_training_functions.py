import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch import optim
from torch.utils.data import DataLoader

from PyTorch.Datasets.Datasets import ImageSuperResDataset
import PyTorch.util.helper_functions as helper
from PyTorch.Models.CnnModules import TwoLayerCNN
from PyTorch.Models.SRCNN import SRCNN
from PyTorch.util.data_augmentation import RandomCropIsr
from PyTorch.util.training_functions import train_classification_model, train_regression_model


def test_train_classification():
    transforms = T.Compose([T.ToTensor()])

    training_data = torchvision.datasets.CIFAR100('data', train=True, download=True, transform=transforms)
    val_data = torchvision.datasets.CIFAR100('data', train=False, download=True, transform=transforms)

    idx = torch.arange(1000)
    training_data = torch.utils.data.Subset(training_data, idx)
    val_data = torch.utils.data.Subset(val_data, idx)

    train_dataloader = DataLoader(training_data, batch_size=32, shuffle=False)
    val_dataloader = DataLoader(val_data, batch_size=32, shuffle=False)

    model = TwoLayerCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print('Training NN')
    train_classification_model(model, criterion, optimizer, train_dataloader, val_dataloader, num_epochs=3)


def test_train_regression():
    IMG_SIZE = (96, 96)
    lr_path, hr_path = helper.download_and_unzip_div2k('data')
    random_crop = RandomCropIsr(IMG_SIZE[0])
    data = ImageSuperResDataset(lr_path, hr_path, transform=random_crop, ds_length=100)

    train_dataloader = DataLoader(data, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(data, batch_size=16, shuffle=True)

    model = SRCNN()
    criterion = nn.L1Loss()

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_regression_model(model, criterion, optimizer, train_dataloader, val_dataloader, num_epochs=3)


if __name__ == '__main__':
    test_train_regression()
    test_train_classification()
