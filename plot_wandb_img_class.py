import os
import wandb
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from tqdm import tqdm


def fetch_runs(username, project_name):
    """Fetch all runs from a W&B project."""
    api = wandb.Api()
    return api.runs(f"{username}/{project_name}")


def group_runs_by_config(runs, exp_type):
    """Group runs by their configuration."""
    grouped_runs = defaultdict(list)

    for run in runs:
        label = generate_legend_label(run.config, exp_type)
        grouped_runs[label].append(run)

    return grouped_runs


def calculate_mean_std(runs, metric):
    """Calculate mean and std for a given metric across runs."""
    metric_values = [run.history()[metric].values for run in runs if metric in run.history()]

    if not metric_values:
        return None, None

    max_len = max(len(values) for values in metric_values)

    # Pad the shorter lists with NaN so that we can compute mean and std
    for values in metric_values:
        while len(values) < max_len:
            values.append(np.nan)

    metric_np = np.array(metric_values)
    mean = np.nanmean(metric_np, axis=0)
    std = np.nanstd(metric_np, axis=0)

    return mean, std


def generate_legend_label(config, exp_type):
    if exp_type == 'arch':
        return f"{config['layer_1'][0].upper()}-{config['layer_2'][0].upper()}-{config['layer_3'][0].upper()}"
    elif exp_type == 'strat':
        label = []
        label.append('four factor') if config['four_factor'] else ''
        label.append('first elem') if config['first_elem_trainable'] else ''
        label.append('bias') if config['deconv_bias'] else ''

        if len(label) == 0:
            return 'none'

        return '-'.join(label)
    else:
        raise NameError('Experiment Type not supported')


def plot_mean_std(username, project_name, metrics: list, exp_type):
    plt.figure(figsize=(20, 8))
    for i, metric in enumerate(metrics):
        print(f'Processing plot {i + 1}')
        plt.subplot(1, 2, i + 1)

        print('Fetching Runs...')
        runs = fetch_runs(username, project_name)

        print('Grouping Runs...')
        grouped_runs = group_runs_by_config(runs, exp_type)

        print('Plotting...')
        for label, runs in tqdm(grouped_runs.items()):

            mean, std = calculate_mean_std(runs, metric)
            if mean is None:
                continue

            epochs = range(1, len(mean) + 1)

            plt.plot(epochs, mean, label=label)
            plt.fill_between(epochs, mean - std, mean + std, alpha=0.2)

        plt.xlabel('Epochs')
        plt.ylabel(metric.replace('_', ' ').capitalize())

        if exp_type == 'arch':
            plt.legend(loc='best', fontsize=15)
        elif exp_type == 'strat':
            plt.legend(loc='upper right', fontsize=10)

    # plt.show()
    os.makedirs('gen_imgs', exist_ok=True)
    plt.savefig(f'gen_imgs/{project_name}_{metrics[0]}_{metrics[1]}.png')
    plt.close()


# Usage
def main():
    username = "viibrem"
    project_name = "Cifar10_strat_09-14"
    exp_type = project_name.split('_')[1]
    metrics = ['train_epoch_loss', 'valid_epoch_loss']
    # metrics = ['train_accuracy', 'valid_accuracy']
    plot_mean_std(username, project_name, metrics=metrics, exp_type=exp_type)


if __name__ == '__main__':
    main()
