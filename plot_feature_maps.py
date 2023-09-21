import torch
import torchvision
import yaml
import wandb
import matplotlib.pyplot as plt
import torchvision.transforms as T

from PyTorch.Models.CnnModules import LeNet5
from PyTorch.util.evaluation_functions import load_weights
from PyTorch.util.impulse_response import impulse_response_of_model, save_tensor_images, image_response_of_model
from plot_wandb_img_class import crop_image


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


def convert_tensor_to_list(tensor):
    tensor = tensor.squeeze()
    list_of_tensors = []

    for i in range(tensor.size(0)):
        list_of_tensors.append(tensor[i])

    return list_of_tensors


def plot_tensors(tensors, v=None, file_name=None):
    # Create a Matplotlib figure with 2x3 subplots (2 rows, 3 columns)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # Loop through the tensors
    for i, tensor in enumerate(tensors):
        row = i // 3  # Calculate the row index
        col = i % 3  # Calculate the column index

        # Convert the PyTorch tensor to a NumPy array
        data = tensor.detach().numpy()

        # Create a color plot (imshow) with the specified color map
        if v is not None:
            im = axes[row, col].imshow(data, cmap='Greys_r', vmin=v[0], vmax=v[1])
        else:
            im = axes[row, col].imshow(data, cmap='Greys_r')

        # Set axis labels and titles for each subplot
        axes[row, col].set_title(f'Filter {i + 1}')
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])

    # Add a single color bar to the figure
    cbar = plt.colorbar(im, ax=axes.ravel().tolist())
    cbar.set_label('Intensity')
    if file_name is None:
        # Show the plot
        plt.show()
    else:
        plt.savefig(file_name)
        plt.close()


def plot_feature_maps():
    model_url = 'viibrem/Cifar10_strat_09-14/runs/fzpjppsx'
    model = download_img_classification_model_from_wandb(model_url)
    mag, phase, y = impulse_response_of_model(model, (3, 32, 32))
    save_tensor_images(mag, 'mag', 'data/test_feature_maps')
    save_tensor_images(y, 'y', 'data/test_feature_maps')
    base = 0
    print(y[0][0][base:base + 5][base:base + 5])
    print('Done')


def compute_and_plot_impulse_responses():
    model_url = 'viibrem/Cifar10_strat_09-14/runs/fzpjppsx'
    model = download_img_classification_model_from_wandb(model_url)
    mag, phase, y = impulse_response_of_model(model, (3, 32, 32))

    outputs = {
        'mag': mag,
        'phase': phase,
        'y': y,
    }
    for key, item in outputs.items():
        tensor = convert_tensor_to_list(item)

        file_name = f'gen_imgs/impulse_reponse-lenet-deconv-{key}.png'
        if key != 'mag':
            plot_tensors(tensor, (-1, 1), file_name)
        else:
            plot_tensors(tensor, (0, 5), file_name)

        crop_image(image_path=file_name, crop_width=130, crop_height=60)
        print(f'Saved figure to {file_name}')


def compute_and_plot_cifar_responses():
    model_url = 'viibrem/Cifar10_strat_09-14/runs/fzpjppsx'
    model = download_img_classification_model_from_wandb(model_url)

    data = torchvision.datasets.CIFAR10('data', train=True, download=True, transform=T.ToTensor())
    image, label = data[29]
    # for i, (image, label) in enumerate(data):
    #     print(i, label)
    #     if i == 50:
    #         break

    y = image_response_of_model(model, image.unsqueeze(0))

    tensor = convert_tensor_to_list(y)

    file_name = f'gen_imgs/cifar-lenet-deconv-y.png'
    plot_tensors(tensor, (-2, 2), file_name)

    crop_image(image_path=file_name, crop_width=130, crop_height=60)
    print(f'Saved figure to {file_name}')


if __name__ == '__main__':
    compute_and_plot_impulse_responses()
    compute_and_plot_cifar_responses()

