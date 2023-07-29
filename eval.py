import argparse

import numpy as np
import yaml
import pandas as pd
import torch
import torchvision

import wandb
from torch import nn
from PIL import Image
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image

from PyTorch.Models.ResNet import ResNet
from PyTorch.Models.SRCNN import SRCNN, BicubicInterpolation
from PyTorch.Models.LossModules import MSE_WITH_DCT, SSIM

import PyTorch.util.helper_functions as helper
from PyTorch.util.data_augmentation import RandomCropIsr, UnpadIsr, PadIsr
from PyTorch.util.evaluation_functions import load_weights, evaluate_regression_model

from PyTorch.Datasets.Datasets import Div2k, IsrEvalDatasets

IMG_SIZE = (96, 96)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-path", "--path", default='data',
                        help="Path to DIV2K. The dataset will be downloaded if not"
                             "located in this path")
    parser.add_argument("-m", "--model_path", default='saved_models/04-14_12-11_srcnn_L1_deconv',
                        help="Path to DIV2K. The dataset will be downloaded if not"
                             "located in this path")
    parser.add_argument("-l", "--loss", default='L1', help="Metric to evaluate loss on ")

    parser.add_argument("--multiple", default='', help="Run with multiple evaluations. "
                                                       "Csv file must be provided with all"
                                                       "filenames in a single column")

    args = parser.parse_args()

    lr_path, hr_path = helper.download_and_unzip_div2k(args.path)

    if args.multiple == '':
        experiments = [{'model_path': args.model_path}]
    else:
        # TODO implement for different losses
        experiments = pd.read_csv(args.multiple, names=['model_path']).to_dict('records')

    for experiment in experiments:
        run_evaluation(lr_path, hr_path, experiment['model_path'], args.loss)


def run_evaluation(lr_path, hr_path, model_path, loss):
    model_path, model_file_name = model_path.split('/')
    model_name_split = model_file_name.split('_')
    model_name = model_name_split[-3]

    deconv = True if model_name_split[-1] == 'deconv' else False

    random_crop = RandomCropIsr(IMG_SIZE[0], train=False)
    data = Div2k(lr_path, hr_path, transform=random_crop)
    dataloader = DataLoader(data, batch_size=16, shuffle=False)

    if model_name == 'srcnn':
        model = SRCNN(deconv=deconv)
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

    model = load_weights(model, f'{model_path}/{model_file_name}')

    evaluate_regression_model(model, criterion, dataloader, name=model_file_name)


def super_resolve_patches(input_images, model, patch_height, patch_width, scale):
    batch_size, channels, height, width = input_images.shape

    # Reshape images into patches
    patches = input_images.unfold(2, patch_height, patch_height).unfold(3, patch_width, patch_width)
    patches = patches.contiguous().view(-1, channels, patch_height, patch_width)

    # Pass patches through model
    output_patches = model(patches)

    # Reshape output patches back into images
    output_images = output_patches.view(batch_size, channels, height // patch_height, width // patch_width,
                                        scale * patch_height, scale * patch_width)
    output_images = output_images.permute(0, 1, 2, 4, 3, 5).contiguous().view(batch_size, channels, scale * height,
                                                                              scale * width)

    return output_images


def eval_on_ds(model: nn.Module, ds_name='Set5', transforms=None, rgb=True, trim_padding=True):
    ds_path = helper.download_and_unzip_sr_ds(ds_name=ds_name)
    data = IsrEvalDatasets(ds_path, rgb=rgb, transform=transforms)
    dataloader = DataLoader(data, batch_size=1, shuffle=False)
    unpad_transform = None

    if trim_padding:
        for transform in transforms:
            if isinstance(transform, PadIsr):
                unpad_transform = UnpadIsr(transform)

    running_loss = {
        'L1': 0.0,
        'MSE': 0.0,
    }
    loss_fns = {
        'L1': nn.L1Loss(),
        'MSE': nn.MSELoss(),
    }

    y_preds = []

    for x, y in dataloader:

        with torch.no_grad():
            y_pred = model(x)
        if unpad_transform is not None:
            y_pred = unpad_transform(y_pred)
            y = unpad_transform(y)

        y_preds += [y_pred.squeeze()]

        for loss_name, loss_fn in loss_fns.items():
            running_loss[loss_name] += loss_fn(y_pred, y).item() / len(dataloader)

    return running_loss, y_preds


def download_model_from_wandb(run_path="viibrem/SuperRes-3LayerCNN-v2/8bb45qep"):
    api = wandb.Api()

    # Retrieve the run
    run = api.run(run_path)

    file_name = None

    # get file names
    files = run.files()

    for f in files:
        if f.name.endswith('.pt'):
            file_name = f.name
            break

    # Download the file
    if file_name is not None:
        file = run.file(file_name)
        file.download(replace=True)

        # download config
        file = run.file('config.yaml')
        file.download(root='saved_models', replace=True)
        with open("saved_models/config.yaml", "r") as stream:
            config = yaml.safe_load(stream)
        model = load_model_with_config(config, file_name)
        return model
    else:
        raise ValueError('Pytorch model was not found in run')


def load_model_with_config(config, model_path):
    model_name = config['model_name']['value']
    if model_name == 'srcnn':
        model = SRCNN(
            num_channels=3 if config['rgb']['value'] else 1,
            deconv=config['deconv']['value'],
            first_elem_trainable=config['first_elem_trainable']['value'],
            bias=config['bias']['value'],
        )
        model = load_weights(model, model_path)
    else:
        raise ValueError(f'{model_name} not supported not supported')
    return model


def display_images(images_tensor):
    if len(images_tensor) % 5 != 0:
        raise ValueError(f"List should be a multiple of 5. List length: {len(images_tensor)}")

    fig, axes = plt.subplots(5, len(images_tensor) // 5, figsize=(10, 12))

    for i, ax in enumerate(axes.flatten()):
        image = images_tensor[i].permute(1, 2, 0).cpu().numpy()
        ax.imshow(image)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def display_pil_images(image_list):
    if len(image_list) % 5 != 0:
        raise ValueError(f"List should be a multiple of 5. List length: {len(image_list)}")

    fig, axes = plt.subplots(5, len(image_list) // 5, figsize=(10, 12))

    for i, ax in enumerate(axes.flatten()):
        image = image_list[i]
        ax.imshow(image)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def eval_model(model):
    running_losses = eval_on_ds(model, 'BSD100')

    print('SRCNN')
    print(running_losses)
    print('===========================================================')

    bicubic = BicubicInterpolation()
    running_losses = eval_on_ds(bicubic, 'BSD100')
    print('Bicubic')
    print(running_losses)
    print('===========================================================')


def load_and_eval_model(run_id):
    model = download_model_from_wandb(run_id)
    running_loss, y_preds = eval_on_ds(model, ds_name='Set5', rgb=True, trim_padding=False)
    return running_loss, y_preds


if __name__ == '__main__':
    load_and_eval_model("viibrem/SuperRes-3LayerCNN-conv-dev/xksdm60k")
