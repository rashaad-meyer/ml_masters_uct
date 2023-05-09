import json
import wandb
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
        wandb.login(key='2121a1ed327903622f934980ca216233453408a0')

        configs = read_json_objects('experiment_csv/classification.txt')

        transforms = T.Compose([T.ToTensor()])

        training_data = torchvision.datasets.CIFAR100('data', train=True, download=True, transform=transforms)
        train_dataloader = DataLoader(training_data, batch_size=64, shuffle=False)

        for hyperparams in configs:
            with wandb.init(project="Cifar100-TwoLayerCNN", config=hyperparams):
                config = wandb.config
                model = TwoLayerCNN(**config)

                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)

                history = train_classification_model(model, criterion, optimizer, train_dataloader, num_epochs=10)

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


def read_json_objects(file_path):
    json_objects = []

    with open(file_path, 'r', encoding='UTF-8') as file:
        lines = file.readlines()

    for line in lines:
        try:
            json_object = json.loads(line.strip())
            json_objects.append(json_object)
        except json.JSONDecodeError:
            print(f"Invalid JSON object at line {len(json_objects) + 1}: {line.strip()}")

    return json_objects


if __name__ == '__main__':
    main()
