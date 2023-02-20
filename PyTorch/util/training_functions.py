import torch
import torch.nn as nn
from tqdm import tqdm


def train_classification_model(model: nn.Module, criterion, optimizer, dataloader, num_epochs=3):
    """
    Trains NN classifier on classification dataset
    :param model: The NN model that you would like to train must be of type nn.Module
    :param criterion: The loss function that you would like to use
    :param optimizer: The optimizer that will optimize the NN
    :param dataloader: Dataloader that loads data from classification dataset
    :param num_epochs: Number of times you want to train the data over
    :return: A dictionary containing training loss and accuracy for each epoch
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    history = {'loss': [], 'accuracy': []}
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1:2d}/{num_epochs}')

        model.train()
        running_loss = 0.0
        running_correct = 0
        data_len = 0

        for X, labels in tqdm(dataloader):
            X = X.to(device)
            labels = labels.to(device)

            with torch.set_grad_enabled(True):
                outputs = model(X)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            running_correct += torch.sum(preds == labels.data).item()
            data_len += X.size(0)

        epoch_loss = running_loss
        epoch_acc = running_correct / data_len

        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_acc)
        print('Loss: {:.4f}, Acc: {:.3f}'.format(epoch_loss, epoch_acc))

    return history


def train_regression_model(model: nn.Module, criterion, optimizer, dataloader, num_epochs=3):
    """
        Trains NN regression on datasets for deblurring and super image resolution
        :param model: The NN model that you would like to train must be of type nn.Module
        :param criterion: The loss function that you would like to use
        :param optimizer: The optimizer that will optimize the NN
        :param dataloader: Dataloader that loads data from classification dataset
        :param num_epochs: Number of times you want to train the data over
        :return: A dictionary containing training loss for each epoch
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    history = {'loss': []}
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for X, y in tqdm(dataloader):
            X = X.to(device)
            y = y.to(device)

            with torch.set_grad_enabled(True):
                outputs = model(X)
                loss = criterion(outputs, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            running_loss += loss.item()

        history['loss'].append(running_loss)

        print('Epoch {:04d} loss: {:.5f}'.format(epoch + 1, running_loss))

    return history
