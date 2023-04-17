import argparse

import pandas as pd

from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader

from PyTorch.Models.ResNet import ResNet
from PyTorch.Models.SRCNN import SRCNN
from PyTorch.Models.LossModules import MSE_WITH_DCT, SSIM

import PyTorch.util.helper_functions as helper
from PyTorch.util.data_augmentation import RandomCropIsr
from PyTorch.util.evaluation_functions import load_weights, evaluate_regression_model

from PyTorch.Datasets.Datasets import ImageSuperResDataset

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


# BigNoob
def run_evaluation(lr_path, hr_path, model_path, loss):
    model_path, model_file_name = model_path.split('/')
    model_name_split = model_file_name.split('_')
    model_name = model_name_split[-3]

    deconv = True if model_name_split[-1] == 'deconv' else False

    random_crop = RandomCropIsr(IMG_SIZE[0], train=False)
    data = ImageSuperResDataset(lr_path, hr_path, transform=random_crop)
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

    model = load_weights(model, model_file_name, model_path)

    evaluate_regression_model(model, criterion, dataloader, name=model_file_name)


if __name__ == '__main__':
    main()
