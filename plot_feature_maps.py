import torch
import yaml
import wandb

from PyTorch.Models.CnnModules import LeNet5
from PyTorch.util.evaluation_functions import load_weights
from PyTorch.util.impulse_response import impulse_response_of_model, save_tensor_images


def download_img_classification_model_from_wandb(run_path):
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
    model = LeNet5(
        layer_1=config['layer_1']['value'],
        layer_2=config['layer_2']['value'],
        layer_3=config['layer_3']['value'],
        deconv_bias=config['deconv_bias']['value'],
        four_factor=config['four_factor']['value'],
        first_elem_trainable=config['first_elem_trainable']['value'],
        input_size=(3, 32, 32)
    )
    model = load_weights(model, model_path)
    return model


def plot_feature_maps():
    model_url = 'viibrem/Cifar10_strat_09-14/runs/fzpjppsx'
    model = download_img_classification_model_from_wandb(model_url)
    mag, phase, y = impulse_response_of_model(model, (3, 32, 32))
    save_tensor_images(mag, 'mag', 'data/test_feature_maps')
    save_tensor_images(y, 'y', 'data/test_feature_maps')
    base = 0
    print(y[0][0][base:base + 5][base:base + 5])
    print('Done')


if __name__ == '__main__':
    plot_feature_maps()
