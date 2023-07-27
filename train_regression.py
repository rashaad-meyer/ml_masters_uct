import argparse

import wandb
import pandas as pd

from torch import nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader

from PyTorch.Models.ResNet import ResNet
from PyTorch.Models.SRCNN import SRCNN
from PyTorch.Models.LossModules import MSE_WITH_DCT, SSIM

import PyTorch.util.helper_functions as helper
from PyTorch.util.data_augmentation import RandomCropIsr, PadIsr
from PyTorch.util.training_functions import train_regression_model

from PyTorch.Datasets.Datasets import Div2k
from eval import eval_on_ds

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

    parser.add_argument('--rgb', dest='rgb', action='store_true')

    args = parser.parse_args()
    wandb.login()

    if args.multiple == '':
        experiments = [{
            'path': args.path,
            'model_name': args.model,
            'deconv': args.deconv,
            'loss': args.loss,
            'num_epochs': args.num_epochs,
            'learning_rate': 10 ** (-args.learning_rate),
            'bias': True,
            'first_elem_trainable': True,
            'rgb': args.rgb
        }]
    else:
        experiments = pd.read_csv(args.multiple, dtype={'deconv': bool}).to_dict('records')

    for experiment in experiments:
        experiment.update({
            'rgb': args.rgb,
        })
        with wandb.init(project="SuperRes-3LayerCNN-conv-dev", config=experiment):
            config = wandb.config
            run_experiment(**config)


def run_experiment(path, model_name, deconv, loss, num_epochs, learning_rate, bias=True, first_elem_trainable=False,
                   rgb=False):
    lr_train_path, hr_train_path = helper.download_and_unzip_div2k(path)
    lr_val_path, hr_val_path = helper.download_and_unzip_div2k(path, dataset_type='valid')

    print('Preparing Dataloader...')
    train_transforms = [RandomCropIsr(IMG_SIZE[0])]
    val_transforms = [RandomCropIsr(256, train=False)]

    if deconv:
        train_transforms += [PadIsr(10)]
        val_transforms += [PadIsr(10)]

    train_data = Div2k(lr_train_path, hr_train_path, rgb=rgb, transform=train_transforms)
    train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True)

    val_data = Div2k(lr_val_path, hr_val_path, rgb=rgb, transform=val_transforms)
    val_dataloader = DataLoader(val_data, batch_size=16, shuffle=True)

    if model_name == 'srcnn':
        num_channels = 3 if rgb else 1
        model = SRCNN(num_channels=num_channels, channels_1=64, channels_2=32, deconv=deconv, bias=bias,
                      first_elem_trainable=first_elem_trainable)
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
    print(f'Set image type to {"RGB" if rgb else "grayscale"}')

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model_file_name = f'{model_name}_{loss}_{"deconv" if deconv else "conv"}'

    print(f'Training for {num_epochs} epochs...')
    history = train_regression_model(model, criterion, optimizer, train_dataloader, val_dataloader,
                                     num_epochs=num_epochs, name=model_file_name)
    if deconv:
        eval_transforms = [PadIsr(IMG_SIZE[0] // 4)]
    else:
        eval_transforms = None

    print('Evaluating on Set5')
    eval_loss, y_preds = eval_on_ds(model, ds_name='Set5', transforms=eval_transforms, rgb=rgb, trim_padding=deconv)

    wandb.log({"Set5 prediction": [wandb.Image(image) for image in y_preds]})

    try:
        wandb.log({"set5_loss": eval_loss})
    except:
        print('Something went wrong when saving set5 loss when')

    print('======================================================================================================\n')


if __name__ == '__main__':
    main()
