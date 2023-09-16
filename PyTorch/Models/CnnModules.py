import torch
import torch.nn as nn
from PyTorch.Models.DeconvModels import Deconv2D


class TwoLayerCNN(nn.Module):
    def __init__(self, img_size=(32, 32), layer_1='conv', layer_2='conv', layer_1_in=3, layer_1_out=32, layer_2_out=32,
                 num_classes=100, deconv_bias=True, first_elem_trainable=True, four_factor=True, dropout=0.5, **kwargs):
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
        self.dropout = nn.Dropout(dropout)

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


class LeNet5(nn.Module):
    def __init__(self, layer_1, layer_2, layer_3='conv', num_classes=10, input_size=(1, 32, 32),
                 deconv_bias=False, first_elem_trainable=False, four_factor=True,
                 channels_1=6, channels_2=16, channels_3=120, filter_size=5, deconv_filter_size=5, **kwargs):
        super(LeNet5, self).__init__()

        # 1st Convolutional Layer
        if layer_1 == 'conv':
            self.conv1 = nn.Conv2d(input_size[0], channels_1, kernel_size=filter_size, stride=1, padding='same')
        elif layer_1 == 'deconv':
            self.conv1 = Deconv2D(input_size[0], channels_1, kernel_size=(deconv_filter_size, deconv_filter_size),
                                  bias=deconv_bias, first_elem_trainable=first_elem_trainable, four_factor=four_factor)
        else:
            raise NameError('Conv1: Proper module not selected')

        self.tanh1 = nn.Tanh()
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        # 2nd Convolutional Layer
        if layer_2 == 'conv':
            self.conv2 = nn.Conv2d(channels_1, channels_2, kernel_size=filter_size, padding='same')
        elif layer_2 == 'deconv':
            self.conv2 = Deconv2D(channels_1, channels_2, kernel_size=(filter_size, filter_size), bias=deconv_bias,
                                  first_elem_trainable=first_elem_trainable, four_factor=four_factor)
        else:
            raise NameError('Conv2: Proper module not selected')

        self.tanh2 = nn.Tanh()
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # 3rd Convolutional Layer
        if layer_3 == 'conv':
            self.conv3 = nn.Conv2d(channels_2, channels_3, kernel_size=filter_size, stride=1, padding='same')
        elif layer_3 == 'deconv':
            self.conv3 = Deconv2D(channels_2, channels_3, kernel_size=(filter_size, filter_size), bias=deconv_bias,
                                  first_elem_trainable=first_elem_trainable, four_factor=four_factor)
        else:
            raise NameError('Conv3: Proper module not selected')

        self.tanh3 = nn.Tanh()

        # Fully Connected Layers
        with torch.no_grad():
            dummy_data = torch.zeros(1, *input_size)
            fc1_input_size = self._forward_features(dummy_data).numel()

        self.fc1 = nn.Linear(fc1_input_size, 84)
        self.tanh4 = nn.Tanh()
        self.fc2 = nn.Linear(84, num_classes)
        self.layer1_out = None

    def _forward_features(self, x):
        x = self.conv1(x)
        self.layer1_out = x

        x = self.tanh1(x)
        x = self.avgpool1(x)

        x = self.conv2(x)
        x = self.tanh2(x)
        x = self.avgpool2(x)

        x = self.conv3(x)
        x = self.tanh3(x)

        return x

    def forward(self, x):
        x = self._forward_features(x)

        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)
        x = self.tanh4(x)
        x = self.fc2(x)

        return x


def test_cnn():
    img = torch.rand((1, 1, 28, 28))
    model = LeNet5('conv', 'conv', 'conv', input_size=img.size()[1:])
    y = model(img)
    print(y.size())


if __name__ == '__main__':
    test_cnn()
