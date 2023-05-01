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
    parser.add_argument("-n", "--num_epochs", default=10, type=int, help="How many epochs to train the network for")
    parser.add_argument("-lr", "--learning_rate", default=3, type=int, help="Learning rate")
    args = parser.parse_args()

    learning_rate = 10 ** -args.learning_rate
    num_epochs = args.num_epochs
    DECONV = args.deconv

    transforms = T.Compose([T.ToTensor()])
    training_data = torchvision.datasets.CIFAR100('data', train=True, download=True, transform=transforms)

    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=False)

    print('Setting up model, loss, and criterion')
    print(f'Use deconv module set to {DECONV}')
    image, label = next(iter(train_dataloader))
    channels = image.size(1)

    model = nn.Sequential(SRCNN(num_channels=channels, deconv=DECONV, use_pixel_shuffle=False),
                          nn.Flatten(), nn.Linear(3072, 100))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print('Training NN')
    history = train_classification_model(model, criterion, optimizer, train_dataloader, num_epochs=num_epochs)

    helper.write_history_to_csv('data', history, 'srcnn', DECONV, '')


if __name__ == '__main__':
    main()
