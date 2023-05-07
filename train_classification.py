import argparse

import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

from PyTorch.Models.CnnModules import TwoLayerCNN
from PyTorch.Models.SRCNN import SRCNN
import PyTorch.util.helper_functions as helper
from PyTorch.util.training_functions import train_classification_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--deconv', dest='deconv', action='store_true')
    parser.add_argument("-n", "--num_epochs", default=10, type=int, help="How many epochs to train the network for")
    parser.add_argument("-lr", "--learning_rate", default=3, type=int, help="Learning rate")
    parser.add_argument('--multi', dest='multi', action='store_true')
    args = parser.parse_args()

    learning_rate = 10 ** -args.learning_rate
    num_epochs = args.num_epochs
    DECONV = args.deconv

    if args.multi:
        experiments = {}
        transforms = T.Compose([T.ToTensor()])

        training_data = torchvision.datasets.CIFAR100('data', train=True, download=True, transform=transforms)
        train_dataloader = DataLoader(training_data, batch_size=64, shuffle=False)

        criterion = nn.CrossEntropyLoss()

        # Two Layer CNN both convolution
        layer_1 = 'conv',
        layer_2 = 'conv',
        layer_1_in = 3,
        layer_1_out = 32,
        layer_2_out = 32
        deconv_bias = False
        first_elem_trainable = False

        model = TwoLayerCNN(
            layer_1=layer_1,
            layer_2=layer_2,
            layer_1_in=layer_1_in,
            layer_1_out=layer_1_out,
            layer_2_out=layer_2_out
        )

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        experiment_name = f'TwoLayerCNN_{layer_1}_{layer_2}_filters_' + \
                          '{layer_1_out}_{layer_1_out}_{deconv_bias}_{first_elem_trainable}'

        experiments[experiment_name] = [model, criterion, optimizer, train_dataloader, num_epochs]

        # ===============================================================================================
        # Two Layer CNN both convolution
        layer_1 = 'deconv',
        layer_2 = 'conv',
        layer_1_in = 3,
        layer_1_out = 32,
        layer_2_out = 32

        model = TwoLayerCNN(
            layer_1=layer_1,
            layer_2=layer_2,
            layer_1_in=layer_1_in,
            layer_1_out=layer_1_out,
            layer_2_out=layer_2_out
        )

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        experiment_name = f'TwoLayerCNN_{layer_1}_{layer_2}_filters_{layer_1_out}_{layer_1_out}'

        experiments[experiment_name] = [model, criterion, optimizer, train_dataloader, num_epochs]

        # ===============================================================================================

        for exp_name, params in experiments:
            history = train_classification_model(*params)

            helper.write_history_to_csv('data', history, exp_name)

    else:
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
