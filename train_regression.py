import argparse

import wandb
import pandas as pd

from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader

from PyTorch.Models.ResNet import ResNet
from PyTorch.Models.SRCNN import SRCNN
from PyTorch.Models.LossModules import MSE_WITH_DCT, SSIM

import PyTorch.util.helper_functions as helper
from PyTorch.util.data_augmentation import RandomCropIsr
from PyTorch.util.training_functions import train_regression_model

from PyTorch.Datasets.Datasets import ImageSuperResDataset

# Global variables
IMG_SIZE = (96, 96)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--path", default='data', help="Path to DIV2K. The dataset will be downloaded if not"
                                                             "located in this path")
    parser.add_argument("-m", "--model", default='srcnn', help="Pick model that you would like to train\n"
                                                               "Options: srcnn,resnet")
    parser.add_argument('--deconv', dest='deconv', action='store_true')
    parser.set_defaults(feature=False)

    parser.add_argument("-l", "--loss", default='L1', help="Which Loss function\n"
                                                           "Options: L1, MSE, DCT, SSIM")
    parser.add_argument("-n", "--num_epochs", default=10, type=int, help="How many epochs to train the network for")
    parser.add_argument("-lr", "--learning_rate", default=4, type=int, help="Learning rate")

    parser.add_argument("--multiple", default='', help="Run with multiple experiments. "
                                                       "Csv file must be provided in the following columns:\n"
                                                       "path, model_name, deconv, loss, num_epochs, learning_rate")

    args = parser.parse_args()
    wandb.login()

    if args.multiple == '':
        experiments = [{'path': args.path,
                        'model_name': args.model,
                        'deconv': args.deconv,
                        'loss': args.loss,
                        'num_epochs': args.num_epochs,
                        'learning_rate': 10 ** (-args.learning_rate),
                        'bias': True,
                        'first_elem_trainable': True}]
    else:
        experiments = pd.read_csv(args.multiple, dtype={'deconv': bool}).to_dict('records')

    for experiment in experiments:
        with wandb.init(project="SuperRes-3LayerCNN", config=experiment):
            config = wandb.config
            run_experiment(**config)


def run_experiment(path, model_name, deconv, loss, num_epochs, learning_rate, bias=True, first_elem_trainable=False):
    lr_train_path, hr_train_path = helper.download_and_unzip_div2k(path)
    lr_val_path, hr_val_path = helper.download_and_unzip_div2k(path, dataset_type='valid')

    print('Preparing Dataloader...')
    random_crop = RandomCropIsr(IMG_SIZE[0])

    train_data = ImageSuperResDataset(lr_train_path, hr_train_path, transform=random_crop)
    train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True)

    val_data = ImageSuperResDataset(lr_val_path, hr_val_path, transform=random_crop)
    val_dataloader = DataLoader(val_data, batch_size=16, shuffle=True)

    if model_name == 'srcnn':
        model = SRCNN(deconv=deconv, bias=bias, first_elem_trainable=first_elem_trainable)
    elif model_name == 'resnet':
        model = ResNet(32, 128)
    else:
        print('Model specified not supported')
        return

    if loss == 'L1':
        criterion = nn.L1Loss()
    elif loss == 'MSE':
        criterion = nn.MSELoss()
    elif loss == 'DCT':
        criterion = MSE_WITH_DCT(IMG_SIZE)
    elif loss == 'SSIM':
        criterion = SSIM()
    else:
        print('Loss function specified not supported')
        return

    print(f'Model set to {model_name}...')
    print(f'Deconv set to {deconv}...')
    print(f'Loss set to {loss}...')
    print(f'Learning rate {learning_rate}...')

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model_file_name = f'{model_name}_{loss}_{"deconv" if deconv else "conv"}'

    print(f'Training for {num_epochs} epochs...')
    history = train_regression_model(model, criterion, optimizer, train_dataloader, val_dataloader,
                                     num_epochs=num_epochs, name=model_file_name)

    print('======================================================================================================\n')


if __name__ == '__main__':
    main()
