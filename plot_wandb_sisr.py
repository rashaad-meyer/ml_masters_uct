import os
import sys
import wandb
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import defaultdict

from plot_wandb_img_class import fetch_runs, crop_image, calculate_mean_std


def generate_legend_label(config, exp_type):
    try:
        return config['legend_label']
    except:
        label = []
        label.append('four factor') if config['four_factor'] else ''
        label.append('first elem') if config['first_elem_trainable'] else ''

        if len(label) == 0:
            return 'none'

        return '-'.join(label)


def group_runs_by_config(runs, exp_type):
    """Group runs by their configuration."""
    grouped_runs = defaultdict(list)

    for run in runs:
        label = generate_legend_label(run.config, exp_type)
        grouped_runs[label].append(run)

    return grouped_runs


def plot_mean_std(username, project_name, baseline_project, additional_project, metrics: list, exp_type):
    plt.figure(figsize=(20, 8))
    for i, metric in enumerate(metrics):
        print(f'Processing plot {i + 1}')
        plt.subplot(1, 2, i + 1)

        print('Fetching Runs...')
        runs = fetch_runs(username, project_name)
        baseline_runs = fetch_runs(username, baseline_project)
        if additional_project is not None:
            additional_runs = fetch_runs(username, additional_project)
        else:
            additional_runs = None

        print('Grouping Runs...')
        grouped_runs = group_runs_by_config(runs, exp_type)

        # Set up the color map
        cmap = plt.get_cmap('tab20b')
        # Create a color normalizer
        norm = mpl.colors.Normalize(vmin=0, vmax=len(grouped_runs))

        print('Plotting...')
        for idx, (label, runs) in tqdm(enumerate(grouped_runs.items())):

            metric_history = runs[0].history()[metric].values
            if metric_history is None:
                continue

            epochs = range(1, len(metric_history) + 1)

            color = cmap(norm(idx))
            plt.plot(epochs, metric_history, label=label, color=color)
            # plt.fill_between(epochs, mean - std, mean + std, alpha=0.2)

        # get first instance
        metric_history = baseline_runs[0].history()[metric].values

        if metric_history is not None:
            epochs = range(1, len(metric_history) + 1)
            plt.plot(epochs, metric_history, label='baseline', color='red')

        if additional_runs is not None:
            # additional project
            metric_history = additional_runs[0].history()[metric].values

            if metric_history is not None:
                epochs = range(1, len(metric_history) + 1)
                plt.plot(epochs, metric_history, label='no padding', color='green')

        plt.xlabel('Epochs', fontsize=16)
        plt.ylabel(metric.replace('_', ' ').capitalize(), fontsize=16)

        plt.legend(loc='lower right', fontsize=15)

    # plt.show()
    os.makedirs('gen_imgs', exist_ok=True)
    file_name = f'gen_imgs/{project_name}_{metrics[0]}_{metrics[1]}.png'
    plt.savefig(file_name)
    plt.close()
    crop_image(image_path=file_name, crop_width=150, crop_height=30)
    print(f'Plot saved to: {file_name}')


# Usage
def main():
    username = "viibrem"
    project_name = "srcnn_padding_x2-09-18"
    baseline_project = "base_srcnn_x2-09-16"
    additional_project = "srcnn_x2-09-17"
    exp_type = project_name.split('_')[1]
    # metrics = ['train_epoch_loss', 'valid_epoch_loss']
    metrics = ['train_psnr', 'valid_psnr']

    plot_mean_std(username, project_name, baseline_project, additional_project, metrics=metrics, exp_type=exp_type)


if __name__ == '__main__':
    main()
