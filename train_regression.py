import argparse

import wandb
import pandas as pd

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader

from PyTorch.Models.ResNet import ResNet
from PyTorch.Models.SRCNN import SRCNN, SrCnnPixelShuffle
from PyTorch.Models.LossModules import MSE_WITH_DCT, SSIM

import PyTorch.util.helper_functions as helper
from PyTorch.util.data_augmentation import RandomCropIsr, PadIsr, RgbToYCbCr, RgbToGrayscale
from PyTorch.util.training_functions import train_regression_model

from PyTorch.Datasets.Datasets import Div2k, NinetyOneImageDataset, EvalDataset, IsrEvalDatasets, H5Dataset
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

    parser.add_argument("--color", default='ycbcr', help="Select color space rgb/gray/ycbcr")

    parser.add_argument("--ds", default='91-image', help="Select dataset div2k/91-image")
    parser.add_argument("-s", "--scale", default=2, type=int, help="Upsampling scale")
    parser.add_argument("--same_size", default='', dest='same_size', action='store_true')
    parser.add_argument("-li", "--log_int", default=0, type=int, help="Log interval: After how many batches "
                                                                      "should metrics be logged")

    args = parser.parse_args()
    wandb.login()

    if not args.multi:
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
            'same_size': args.same_size,
            'log_interval': args.log_int,
        }]
    else:
        experiments = pd.read_csv('experiment_csv/regression.csv',
                                  dtype={'deconv': bool}).to_dict('records')

    for experiment in experiments:
        experiment.update({
            'color': args.color,
            'dataset': args.ds,
            'scale': args.scale,
            'same_size': args.same_size,
            'log_interval': args.log_int,
            'batch_norm': 'before tanh'
        })
        with wandb.init(project=f"SRCNN-x{args.scale}-batchnorm-v01", config=experiment):
            config = wandb.config
            run_experiment(**config)


def run_experiment(path, model_name, deconv, loss, num_epochs, learning_rate, bias=True, first_elem_trainable=False,
                   color='rgb', dataset='div2k', scale=2, padding=False, same_size=False, pad_inner=None,
                   four_factor=True, log_interval=0):
    # FIXME add padding to list of arguments
    torch.manual_seed(42)
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

        g = torch.Generator()
        g.manual_seed(42)

        train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True, generator=g)
        val_dataloader = DataLoader(val_data, batch_size=16, shuffle=True, generator=g)

    elif dataset == '91-image':
        ds_path = helper.download_91_image_and_set5_ds('data/sr/srcnn', scale=scale)

        train_path = ds_path['91-image']
        val_path = ds_path['Set5']

        train_data = NinetyOneImageDataset(train_path, same_size=same_size)
        val_data = EvalDataset(val_path, same_size=same_size)

        g = torch.Generator()
        g.manual_seed(0)

        train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True, generator=g)
        val_dataloader = DataLoader(val_data, batch_size=1, shuffle=True, generator=g)

    elif dataset == 'h5':
        train_path = 'data/h5/div2k_x2_ycbcr_32.h5'
        helper.download_div2k_h5(train_path)

        val_path = helper.download_and_unzip_sr_ds(path='data/sr', ds_name='Set5')

        val_transforms = [RgbToYCbCr(return_only_y=True)]

        train_data = H5Dataset(train_path)
        val_data = IsrEvalDatasets(val_path, transform=val_transforms)

        g = torch.Generator()
        g.manual_seed(0)

        train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True, generator=g)
        val_dataloader = DataLoader(val_data, batch_size=1, shuffle=True, generator=g)

    else:
        raise ValueError('Dataset specified not supported choose div2k or 91-image')

    if model_name == 'srcnn':

        if dataset == '91-image':
            num_channels = 1
            use_pixel_shuffle = not same_size
        elif dataset == 'h5':
            num_channels = 1
            use_pixel_shuffle = True
        else:
            num_channels = 3 if color == 'rgb' else 1
            use_pixel_shuffle = True

        if use_pixel_shuffle:
            model = SrCnnPixelShuffle(num_channels=num_channels, channels_1=64, channels_2=32, deconv=deconv, bias=bias,
                                      first_elem_trainable=first_elem_trainable, pad_inner=pad_inner,
                                      four_factor=four_factor, upscale_factor=scale)
        else:
            model = SRCNN(num_channels=num_channels, channels_1=64, channels_2=32, deconv=deconv, bias=bias,
                          first_elem_trainable=first_elem_trainable, use_pixel_shuffle=use_pixel_shuffle,
                          pad_inner=pad_inner, four_factor=four_factor, upscale_factor=scale)
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
        criterion = MSE_WITH_DCT('l1')
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
    print(f'Inner padding (deconv) set to {pad_inner}')
    if dataset == '91-image':
        print('LR are being upscaled before being passed to network')

    if model_name == 'srcnn':
        optimizer = optim.AdamW([
            {'params': model.conv1.parameters()},
            {'params': model.conv2.parameters()},
            {'params': model.conv3.parameters(), 'lr': learning_rate * 0.1}
        ], lr=learning_rate)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model_file_name = f'{model_name}_{loss}_{"deconv" if deconv else "conv"}'

    print(f'Training for {num_epochs} epochs...')
    history = train_regression_model(model, criterion, optimizer, train_dataloader, val_dataloader,
                                     num_epochs=num_epochs, name=model_file_name, log_interval=log_interval)
    if deconv and padding:
        eval_transforms = [PadIsr(IMG_SIZE[0] // 4)]
    else:
        eval_transforms = []

    if color == 'ycbcr' or dataset == 'h5':
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
