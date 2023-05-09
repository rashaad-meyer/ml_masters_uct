import torch.nn as nn
from PyTorch.Models.DeconvModels import Deconv2DMultiFilter


class TwoLayerCNN(nn.Module):
    def __init__(self, img_size=(32, 32), layer_1='conv', layer_2='conv', layer_1_in=3, layer_1_out=32, layer_2_out=32,
                 num_classes=100, deconv_bias=True, first_elem_trainable=True):
        super(TwoLayerCNN, self).__init__()

        if layer_1 == 'conv':
            self.layer1 = nn.Conv2d(layer_1_in, layer_1_out, kernel_size=3, padding='same')
        elif layer_1 == 'deconv':
            self.layer1 = Deconv2DMultiFilter(layer_1_in, layer_1_out, kernel_size=3, bias=deconv_bias,
                                              first_elem_trainable=first_elem_trainable)

        if layer_2 == 'conv':
            self.layer2 = nn.Conv2d(layer_1_out, layer_2_out, kernel_size=3, padding='same')
        elif layer_2 == 'deconv':
            self.layer2 = Deconv2DMultiFilter(layer_1_out, layer_2_out, kernel_size=3, bias=deconv_bias,
                                              first_elem_trainable=first_elem_trainable)

        self.fc1 = nn.Linear(layer_2_out * img_size[0] * img_size[1], num_classes)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.flatten(x)
        print(x.size())
        x = self.fc1(x)
        return x
