import argparse

import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

from PyTorch.Models.SRCNN import SRCNN
import PyTorch.util.helper_functions as helper
from PyTorch.util.training_functions import train_classification_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--deconv', dest='deconv', action='store_true')
    args = parser.parse_args()

    DECONV = args.deconv

    transforms = T.Compose([T.ToTensor()])
    training_data = torchvision.datasets.CIFAR100('data', train=True, download=True, transform=transforms)

    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=False)
    image, label = next(iter(train_dataloader))
    print(label.size())
    channels = image.size(1)

    model = nn.Sequential(SRCNN(num_channels=channels, deconv=DECONV, use_pixel_shuffle=False),
                          nn.Linear(3072, 10), )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    history = train_classification_model(model, criterion, optimizer, train_dataloader, num_epochs=10)

    helper.write_history_to_csv('data', history, 'srcnn', DECONV, '')


if __name__ == '__main__':
    main()
