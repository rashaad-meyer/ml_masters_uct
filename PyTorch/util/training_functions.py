import torch
import torch.nn as nn
from tqdm import tqdm
import time
import os
from datetime import datetime


def train_classification_model(model: nn.Module, criterion, optimizer, dataloader, num_epochs=3, deconv=False):
    """
        Trains NN classifier on classification dataset
        :param model: The NN model that you would like to train must be of type nn.Module
        :param criterion: The loss function that you would like to use
        :param optimizer: The optimizer that will optimize the NN
        :param dataloader: Dataloader that loads data from classification dataset
        :param num_epochs: Number of times you want to train the data over
        :param deconv: boolean check whether to save deconv kernel or not
        :return: A dictionary containing training loss and accuracy for each epoch
        and list of the kernels after each epoch if the model is a deconv layer
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = criterion.to(device)
    if num_epochs > 20:
        print_epoch_every = num_epochs // 20
    else:
        print_epoch_every = 1
    history = {'loss': [], 'accuracy': [], 'time': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1:2d}/{num_epochs}')

        model.train()
        running_loss = 0.0
        running_correct = 0
        data_len = 0
        start = time.time()

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
        end = time.time()

        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_acc)
        history['time'].append(round(end - start))

        if (epoch + 1) % print_epoch_every == 0:
            print('Loss: {:.4f}, Acc: {:.3f}'.format(epoch_loss, epoch_acc))

    return history


def train_regression_model(model: nn.Module, criterion, optimizer, dataloader, num_epochs=3, name='model'):
    """
        Trains NN regression on datasets for deblurring and super image resolution
        :param name: name of model that you will use
        :param model: The NN model that you would like to train must be of type nn.Module
        :param criterion: The loss function that you would like to use
        :param optimizer: The optimizer that will optimize the NN
        :param dataloader: Dataloader that loads data from classification dataset
        :param num_epochs: Number of times you want to train the data over
        :param deconv: boolean check whether to save deconv kernel or not
        :return: A dictionary containing training loss for each epoch
        and list of the kernels after each epoch if the model is a deconv layer
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = criterion.to(device)
    if num_epochs > 20:
        print_epoch_every = num_epochs // 20
    else:
        print_epoch_every = 1
    history = {'loss': [], 'time': []}

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        start = time.time()
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

        end = time.time()

        history['loss'].append(running_loss)
        history['time'].append(round(end - start))

        save_model(model, name, epoch, running_loss)

        if (epoch + 1) % print_epoch_every == 0:
            print('Epoch {:04d} loss: {:.5f}'.format(epoch + 1, running_loss))

    return history


def save_model(model, name, epoch, loss, folder='saved_models'):
    if not os.path.exists(folder):
        os.makedirs(folder)

    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f'{folder}/{dt_string}_{name}_{epoch}'
    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'loss': loss, }, file_name)
