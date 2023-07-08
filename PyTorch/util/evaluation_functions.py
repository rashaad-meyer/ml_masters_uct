import torch
import torch.nn as nn


def evaluate_regression_model(model: nn.Module, criterion, dataloader, name='model'):
    """
        Trains NN regression on datasets for deblurring and super image resolution
        :param model: The NN model that you would like to evaluate must be of type nn.Module
        :param criterion: The loss function that you would like to use
        :param dataloader: Dataloader that loads data from classification dataset
        :param name: name of model that you will use
        :return: returns the loss of the model on the dataset
        and list of the kernels after each epoch if the model is a deconv layer
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = criterion.to(device)

    model.eval()
    running_loss = 0.0
    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(X)
            loss = criterion(outputs, y)

        running_loss += loss.item()

    print(f'{name} loss: {running_loss:.5f}')

    return running_loss


def load_weights(model, model_name, folder='saved_models'):
    model_path = f'{folder}/{model_name}'

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)

    return model
