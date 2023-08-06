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
from PyTorch.util.data_augmentation import RandomCropIsr, PadIsr, RgbToYCbCr, RgbToGrayscale
from PyTorch.util.training_functions import train_regression_model

from PyTorch.Datasets.Datasets import Div2k, TrainDataset, EvalDataset
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

    parser.add_argument("--multi", default='', dest='multi', action='store_true')

    parser.add_argument("--color", default='rgb', help="Select color space rgb/gray/ycbcr")

    parser.add_argument("--ds", default='div2k', help="Select dataset div2k/91-image")
    parser.add_argument("-s", "--scale", default=2, type=int, help="Upsampling scale")

    args = parser.parse_args()
    wandb.login()

    if args.multi:
        experiments = [{
            'path': args.path,
            'model_name': args.model,
            'deconv': args.deconv,
            'loss': args.loss,
            'num_epochs': args.num_epochs,
            'learning_rate': 10 ** (-args.learning_rate),
            'bias': True,
            'first_elem_trainable': True,
            'color': args.color,
            'dataset': args.ds,
            'scale': args.scale,
        }]
    else:
        experiments = pd.read_csv('experiment_csv/regression.csv',
                                  dtype={'deconv': bool}).to_dict('records')

    for experiment in experiments:
        experiment.update({
            'color': args.color,
            'dataset': args.ds,
            'scale': args.scale,
        })
        with wandb.init(project=f"SRCNN-x{args.scale}-dev-v00", config=experiment):
            config = wandb.config
            run_experiment(**config)


def run_experiment(path, model_name, deconv, loss, num_epochs, learning_rate, bias=True, first_elem_trainable=False,
                   color='rgb', dataset='div2k', scale=2, padding=False):
    # FIXME add padding to list of arguments
    if dataset == 'div2k':
        lr_train_path, hr_train_path = helper.download_and_unzip_div2k(path)
        lr_val_path, hr_val_path = helper.download_and_unzip_div2k(path, dataset_type='valid')

        print('Preparing Dataloader...')
        train_transforms = [RandomCropIsr(IMG_SIZE[0])]
        val_transforms = [RandomCropIsr(256, train=False)]

        if color == 'ycbcr':
            train_transforms += [RgbToYCbCr(return_only_y=True)]
            val_transforms += [RgbToYCbCr(return_only_y=True)]
        elif color == 'grayscale':
            train_transforms += [RgbToGrayscale()]
            val_transforms += [RgbToGrayscale()]
        elif color != 'rgb':
            raise ValueError('Incorrect color space specified. Please choose between rgb, ycbcr, and grayscale')

        if deconv and padding:
            train_transforms += [PadIsr(10)]
            val_transforms += [PadIsr(10)]

        train_data = Div2k(lr_train_path, hr_train_path, transform=train_transforms)
        val_data = Div2k(lr_val_path, hr_val_path, transform=val_transforms)

        train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True)
        val_dataloader = DataLoader(val_data, batch_size=16, shuffle=True)

    elif dataset == '91-image':
        ds_path = helper.download_91_image_and_set5_ds('data/sr/srcnn', scale=scale)

        train_path = ds_path['91-image']
        val_path = ds_path['Set5']

        # TODO resizing
        train_data = TrainDataset(train_path, same_size=False)
        val_data = EvalDataset(val_path, same_size=False)

        train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True)
        val_dataloader = DataLoader(val_data, batch_size=1, shuffle=True)

    else:
        raise ValueError('Dataset specified not supported choose div2k or 91-image')

    if model_name == 'srcnn':

        if dataset == '91-image':
            num_channels = 1
            use_pixel_shuffle = True
            # TODO resizing
        else:
            num_channels = 3 if color == 'rgb' else 1
            use_pixel_shuffle = True

        model = SRCNN(num_channels=num_channels, channels_1=64, channels_2=32, deconv=deconv, bias=bias,
                      first_elem_trainable=first_elem_trainable, use_pixel_shuffle=use_pixel_shuffle)
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
    if dataset == 'div2k':
        print(f'Set image type to {color}')
    print(f'Dataset set to {dataset}')

    if model_name == 'srcnn':
        optimizer = optim.Adam([
            {'params': model.conv1.parameters()},
            {'params': model.conv2.parameters()},
            {'params': model.conv3.parameters(), 'lr': learning_rate * 0.1}
            # TODO add other learnable parameters
        ], lr=learning_rate)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model_file_name = f'{model_name}_{loss}_{"deconv" if deconv else "conv"}'

    print(f'Training for {num_epochs} epochs...')
    history = train_regression_model(model, criterion, optimizer, train_dataloader, val_dataloader,
                                     num_epochs=num_epochs, name=model_file_name)
    if deconv and padding:
        eval_transforms = [PadIsr(IMG_SIZE[0] // 4)]
    else:
        eval_transforms = []

    if color == 'ycbcr':
        eval_transforms += [RgbToYCbCr(return_only_y=True)]
    elif color == 'grayscale':
        eval_transforms += [RgbToGrayscale()]
    elif color != 'rgb':
        raise ValueError('Incorrect color space specified. Please choose between rgb, ycbcr, and grayscale')

    try:
        print('Evaluating on Set5')
        eval_loss, y_preds = eval_on_ds(model, ds_name='Set5', transforms=eval_transforms, trim_padding=padding)
        log_predictions_to_wandb(y_preds)
        wandb.log({"set5_loss": eval_loss})
    except Exception as e:
        print(f"Couldn't log predicted images to wandb:\n{e}")

    print('======================================================================================================\n')


def log_predictions_to_wandb(y_preds):
    np_preds = []
    for pred in y_preds:
        np_preds += [pred.permute(1, 2, 0).cpu().numpy()]

    wandb.log({"Set5 prediction": [wandb.Image(image) for image in np_preds]})


if __name__ == '__main__':
    main()
