import torch
import torch.nn as nn
from PyTorch.Models.DeconvModels import Deconv2D


class TwoLayerCNN(nn.Module):
    def __init__(self, img_size=(32, 32), layer_1='conv', layer_2='conv', layer_1_in=3, layer_1_out=32, layer_2_out=32,
                 num_classes=100, deconv_bias=True, first_elem_trainable=True, four_factor=True):
        super(TwoLayerCNN, self).__init__()

        if layer_1 == 'conv':
            self.layer1 = nn.Conv2d(layer_1_in, layer_1_out, kernel_size=3, padding='same')
        elif layer_1 == 'deconv':
            self.layer1 = Deconv2D(layer_1_in, layer_1_out, kernel_size=(3, 3), bias=deconv_bias,
                                   first_elem_trainable=first_elem_trainable, four_factor=four_factor)

        if layer_2 == 'conv':
            self.layer2 = nn.Conv2d(layer_1_out, layer_2_out, kernel_size=3, padding='same')
        elif layer_2 == 'deconv':
            self.layer2 = Deconv2D(layer_1_out, layer_2_out, kernel_size=(3, 3), bias=deconv_bias,
                                   first_elem_trainable=first_elem_trainable, four_factor=True)

        # Adding a max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # Adjust the input size of fc1
        self.fc1 = nn.Linear(layer_2_out * img_size[0] // 2 * img_size[1] // 2, num_classes)

        self.layer1_out = None
        self.layer2_out = None

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.layer1(x)
        self.layer1_out = x
        x = self.relu(x)

        x = self.layer2(x)
        self.layer2_out = x
        x = self.relu(x)
        x = self.pool(x)

        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc1(x)
        return x


class ObjDetCNN(nn.Module):
    def __init__(self, img_size=(448, 448), layer_1='conv', layer_2='conv', layer_1_in=3, layer_1_out=32,
                 layer_2_out=32,
                 num_classes=100, deconv_bias=True, first_elem_trainable=True, four_factor=True):
        super(ObjDetCNN, self).__init__()

        if layer_1 == 'conv':
            self.layer1 = nn.Conv2d(layer_1_in, layer_1_out, kernel_size=3, padding='same')
        elif layer_1 == 'deconv':
            self.layer1 = Deconv2D(layer_1_in, layer_1_out, kernel_size=(3, 3), bias=deconv_bias,
                                   first_elem_trainable=first_elem_trainable, four_factor=four_factor)

        if layer_2 == 'conv':
            self.layer2 = nn.Conv2d(layer_1_out, layer_2_out, kernel_size=3, padding='same')
        elif layer_2 == 'deconv':
            self.layer2 = Deconv2D(layer_1_out, layer_2_out, kernel_size=(3, 3), bias=deconv_bias,
                                   first_elem_trainable=first_elem_trainable, four_factor=True)

        # ensure that conv output size is 7x7
        kernel_size = stride_size = (img_size[0] // 7, img_size[1] // 7)

        self.layer3 = nn.Conv2d(layer_2_out, layer_2_out, kernel_size=kernel_size, stride=stride_size)

        self.fc1 = nn.Linear(layer_2_out * 7 * 7, num_classes)

        self.layer1_out = None
        self.layer2_out = None

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.layer1(x)
        self.layer1_out = x
        x = self.relu(x)

        x = self.layer2(x)
        self.layer2_out = x
        x = self.relu(x)

        x = self.layer3(x)
        x = self.relu(x)

        x = self.flatten(x)
        x = self.fc1(x)
        return x


def test_two_layer_cnn():
    img = torch.rand((1, 3, 32, 32))
    model = TwoLayerCNN()

    y = model(img)

    print(y.size())


if __name__ == '__main__':
    test_two_layer_cnn()
