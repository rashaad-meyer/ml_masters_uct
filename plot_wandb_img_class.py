import json
import wandb
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


def fetch_runs(username, project_name):
    """Fetch all runs from a W&B project."""
    api = wandb.Api()
    return api.runs(f"{username}/{project_name}")


def group_runs_by_config(runs, exp_type):
    """Group runs by their configuration."""
    grouped_runs = defaultdict(list)

    for run in runs:
        label = generate_legend_label(run.config, exp_type)  # Convert the config dict to string to make it hashable
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
        return f"{config['layer_1']}-{config['layer_2']}-{config['layer_3']}"
    else:
        raise NotImplemented('Type still needs to be implemented')


def plot_mean_std(username, project_name, metric, exp_type):
    runs = fetch_runs(username, project_name)
    grouped_runs = group_runs_by_config(runs, exp_type)

    plt.figure(figsize=(12, 8))

    for label, runs in grouped_runs.items():
        mean, std = calculate_mean_std(runs, metric)
        if mean is None:
            continue

        epochs = range(1, len(mean) + 1)

        plt.plot(epochs, mean, label=label)
        plt.fill_between(epochs, mean - std, mean + std, alpha=0.2)

    plt.xlabel('Epochs')
    plt.ylabel(metric)
    plt.legend()
    plt.title(f'{metric} over epochs for various configurations')
    plt.show()


# Usage
def main():
    username = "viibrem"
    project_name = "Cifar-final-dev-v0.2"
    exp_type = 'arch'
    plot_mean_std(username, project_name, metric='valid_accuracy', exp_type=exp_type)


if __name__ == '__main__':
    main()
