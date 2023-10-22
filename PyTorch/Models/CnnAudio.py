import torch.nn as nn

from PyTorch.Models.DeconvModels import Deconv1D


class AudioDNN(nn.Module):
    def __init__(self, mode='deconv'):
        super(AudioDNN, self).__init__()
        if mode == 'deconv':
            self.conv1 = Deconv1D(bias=False, first_elem_trainable=False)
        elif mode == 'conv':
            self.conv1 = nn.Conv1d(1, 1, 1)
        self.fc1 = nn.Linear(89009, 10)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x
