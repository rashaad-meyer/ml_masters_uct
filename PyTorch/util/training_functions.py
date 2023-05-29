import torch
import wandb
import torch.nn as nn
from tqdm import tqdm
import os
from datetime import datetime

from PyTorch.util.impulse_response import impulse_response_of_model, save_tensor_images


def train_classification_model(model: nn.Module, criterion, optimizer, dataloader, num_epochs=3):
    """
        Trains NN classifier on classification dataset
        :param model: The NN model that you would like to train must be of type nn.Module
        :param criterion: The loss function that you would like to use
        :param optimizer: The optimizer that will optimize the NN
        :param dataloader: Dataloader that loads data from classification dataset
        :param num_epochs: Number of times you want to train the data over
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

    wandb.watch(model, criterion, log="all", log_freq=10)

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

        wandb.log({"epoch_loss": epoch_loss, "accuracy": epoch_acc}, step=epoch)

        image, _ = next(iter(dataloader))

        try:
            impulse_responses = impulse_response_of_model(model, image.size())
            response_images = save_tensor_images(impulse_responses)
            wandb.log({f"epoch_{epoch:04d}_impulse": [wandb.Image(image) for image in response_images]})
        except:
            print('First layer is not deconv. Not logging impulse responses')

        now = datetime.now()
        dt_string = now.strftime("%m-%d_%H-%M")
        history['time'].append(dt_string)

        if (epoch + 1) % print_epoch_every == 0:
            print('Loss: {:.4f}, Acc: {:.3f}'.format(epoch_loss, epoch_acc))

    print('======================================================================================================\n')

    return history


def train_regression_model(model: nn.Module, criterion, optimizer, dataloader, num_epochs=3, name='model'):
    """
        Trains NN regression on datasets for deblurring and super image resolution
        :param model: The NN model that you would like to train must be of type nn.Module
        :param criterion: The loss function that you would like to use
        :param optimizer: The optimizer that will optimize the NN
        :param dataloader: Dataloader that loads data from classification dataset
        :param num_epochs: Number of times you want to train the data over
        :param name: name of model that you will use
        :return: A dictionary containing training loss for each epoch
        and list of the kernels after each epoch if the model is a deconv layer
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = criterion.to(device)
    if num_epochs > 200:
        print_epoch_every = num_epochs // 20
    else:
        print_epoch_every = 1
    history = {'loss': [], 'time': []}

    start_time = datetime.now().strftime("%m-%d_%H-%M")
    experiment_name = f'{start_time}_{name}'

    wandb.watch(model, criterion, log="all", log_freq=10)

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

        now = datetime.now()
        dt_string = now.strftime("%m-%d_%H-%M")
        history['time'].append(dt_string)

        wandb.log({"epoch_loss": running_loss}, step=epoch)

        if (epoch + 1) % print_epoch_every == 0:
            print('Epoch {:04d} loss: {:.5f}'.format(epoch + 1, running_loss))

        if epoch == 0 or min(history['loss'][:-1]) > history['loss'][-1]:
            save_model(model, experiment_name, epoch, running_loss)

    wandb.log({"best_loss": min(history['loss'])})

    return history


def save_model(model, name, epoch, loss, folder='saved_models'):
    if not os.path.exists(folder):
        os.makedirs(folder)

    file_name = f'{folder}/{name}'

    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'loss': loss, }, file_name)
    print(f'Model saved at {file_name}')
