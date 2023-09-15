import json
import wandb
import argparse

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

from PyTorch.Models.CnnModules import LeNet5, TwoLayerCNN
from PyTorch.Models.SRCNN import SRCNN
import PyTorch.util.helper_functions as helper
from PyTorch.util.training_functions import train_classification_model
import datetime

# Store it in a variable called DATE
DATE = datetime.datetime.now().strftime("%m-%d")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--deconv', dest='deconv', action='store_true')
    parser.add_argument("-n", "--num_epochs", default=10, type=int, help="How many epochs to train the network for")
    parser.add_argument("-m", "--model", default='lenet', type=str, help="Choose between two model: lenet or twolayer")
    parser.add_argument('--multi', default='experiment_csv/img_class/dev.txt',
                        help="Choose which file the experiments are contained in. "
                             "Set to false if you want to use single")
    parser.add_argument("-d", "--ds", default='cifar', type=str, help="Choose which dataset to use")
    args = parser.parse_args()

    learning_rate = 10 ** -args.learning_rate
    num_epochs = args.num_epochs
    DECONV = args.deconv
    batch_size = args.batch_size

    if args.multi != 'false':
        wandb.login()

        # run each experiment 3 times
        for exp_run in range(3):
            torch.manual_seed(exp_run)
            configs = read_json_objects(args.multi)

            for hyperparams in configs:
                exp_type = args.multi.split('/')[-1][:-4]

                batch_size = hyperparams.get('batch_size', 32)
                learning_rate = hyperparams.get('learning_rate', 0.001)
                optimizer_type = hyperparams.get('optimizer', 'adam').lower()
                exp_type = hyperparams.get('exp_type', exp_type).lower()

                g = torch.Generator()
                g.manual_seed(exp_run)

                train_dataloader, training_data, val_dataloader = get_dataloaders(args, batch_size, g)

                num_classes = len(training_data.classes)

                for key, item in hyperparams.items():
                    print(f'{key} is set to {item}')
                with wandb.init(project=f"Cifar10_{exp_type}_{DATE}", config=hyperparams):
                    config = wandb.config
                    if args.model == 'twolayer':
                        model = TwoLayerCNN(**config, num_classes=num_classes, dropout=0.0)
                    elif args.model == 'lenet':
                        example, _ = training_data[0]
                        model = LeNet5(**config, num_classes=num_classes, input_size=example.size())
                    else:
                        raise NameError('Please pick valid model: lenet or twolayer')

                    criterion = nn.CrossEntropyLoss()

                    if optimizer_type == 'adam':
                        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                    elif optimizer_type == 'sgd':
                        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
                    else:
                        raise NameError('Invalid optimizer specified, please use adam or sgd.')

                    history = train_classification_model(model, criterion, optimizer, train_dataloader, val_dataloader,
                                                         num_epochs=10)

    else:
        transforms = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomCrop(32, padding=4),
            T.ToTensor(),
        ])

        g = torch.Generator()
        g.manual_seed(42)

        training_data = torchvision.datasets.CIFAR100('data', train=True, download=True, transform=transforms)
        train_dataloader = DataLoader(training_data, batch_size=64, shuffle=False, generator=g)

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


def get_dataloaders(args, batch_size, g):
    if args.ds == 'cifar':
        transforms = T.Compose([
            # T.RandomHorizontalFlip(),
            # T.RandomCrop(32, padding=4),
            T.ToTensor(),
        ])
        training_data = torchvision.datasets.CIFAR10('data', train=True, download=True, transform=transforms)
        train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, generator=g)

        val_data = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=T.ToTensor())
        val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    elif args.ds == 'dtd':
        transforms = T.Compose([
            # T.RandomHorizontalFlip(),
            # T.RandomCrop(32, padding=4),
            T.ToTensor(),
            T.Resize((224, 224)),
        ])
        training_data = torchvision.datasets.DTD(root='data', split='train', download=True,
                                                 transform=transforms)
        train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, generator=g)

        val_data = torchvision.datasets.DTD(root='data', split='train', download=True, transform=transforms)
        val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    else:
        raise NameError('Invalid class specified please pick from the following: cifar / dtd')
    return train_dataloader, training_data, val_dataloader


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
