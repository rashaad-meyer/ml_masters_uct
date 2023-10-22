import wandb
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from PyTorch.Datasets.AudioDatasets import UrbanSound8KDataset
from PyTorch.Models.CnnAudio import AudioDNN
from PyTorch.util.training_functions import train_classification_model

EXPERIMENTS = [
    {'mode': 'deconv'},
    {'mode': 'conv'}
]


def main():
    training_data = UrbanSound8KDataset(csv_file='data/urban8k/UrbanSound8K/UrbanSound8K/metadata/UrbanSound8K.csv',
                                        root_dir='data/urban8k/UrbanSound8K/UrbanSound8K/audio/')

    train_dataloader = DataLoader(training_data, batch_size=32, shuffle=True)

    for experiment in EXPERIMENTS:
        with wandb.init(project=f"one-dimension-example", config=experiment):
            config = wandb.config
            model = AudioDNN(**experiment)

            criterion = nn.CrossEntropyLoss()
            optimizer = Adam(model.parameters(), lr=1e-3)

            train_classification_model(model, criterion, optimizer, train_dataloader, num_epochs=10)


if __name__ == '__main__':
    main()
