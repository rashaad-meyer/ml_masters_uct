import torch
import wandb
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import os
from datetime import datetime

from PyTorch.Models.LossModules import PSNR
from PyTorch.util.impulse_response import impulse_response_of_model, save_tensor_images, check_filter_diff


def train_classification_model(model: nn.Module, criterion, optimizer, train_dataloader, valid_dataloader=None,
                               num_epochs=3, name='model'):
    """
        Trains NN classifier on classification dataset
        :param model: The NN model that you would like to train must be of type nn.Module
        :param criterion: The loss function that you would like to use
        :param optimizer: The optimizer that will optimize the NN
        :param train_dataloader: Dataloader that loads training data from classification dataset
        :param valid_dataloader: Dataloader that loads validation data from classification dataset
        :param num_epochs: Number of times you want to train the data over
        :param name: name of the model
        :return: A dictionary containing training and validation loss and accuracy for each epoch
        and list of the kernels after each epoch if the model is a deconv layer
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = criterion.to(device)
    history = {'train_loss': [], 'train_accuracy': [], 'valid_loss': [], 'valid_accuracy': [], 'time': []}

    start_time = datetime.now().strftime("%m-%d_%H-%M")
    experiment_name = f'{start_time}_{name}'
    best_model_path = None

    try:
        wandb.watch(model, criterion, log="all", log_freq=10)
    except:
        print('Something went wrong with wandb')

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1:2d}/{num_epochs}')

        model.train()
        running_loss = 0.0
        running_correct = 0
        data_len = 0

        for X, labels in tqdm(train_dataloader):
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

        epoch_loss = running_loss / data_len
        epoch_acc = running_correct / data_len

        history['train_loss'].append(epoch_loss)
        history['train_accuracy'].append(epoch_acc)

        try:
            wandb.log({"train_epoch_loss": epoch_loss, "train_accuracy": epoch_acc}, step=epoch)
        except:
            print('Something went wrong with wandb')

        print(f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.3f}')

        if valid_dataloader:
            model.eval()
            valid_running_loss = 0.0
            valid_running_correct = 0
            valid_data_len = 0

            for X, labels in valid_dataloader:
                X = X.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    outputs = model(X)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                valid_running_loss += loss.item()
                valid_running_correct += torch.sum(preds == labels.data).item()
                valid_data_len += X.size(0)

            valid_epoch_loss = valid_running_loss / valid_data_len
            valid_epoch_acc = valid_running_correct / valid_data_len

            history['valid_loss'].append(valid_epoch_loss)
            history['valid_accuracy'].append(valid_epoch_acc)

            print(f'Validation Loss: {valid_epoch_loss:.4f}, Validation Acc: {valid_epoch_acc:.3f}')

            try:
                wandb.log({"valid_epoch_loss": valid_epoch_loss, "valid_accuracy": valid_epoch_acc}, step=epoch)
            except:
                print('Something went wrong with wandb')

        if epoch == 0 or min(history['train_loss'][:-1]) > history['train_loss'][-1]:
            best_model_path = save_model(model, experiment_name)

        now = datetime.now()
        dt_string = now.strftime("%m-%d_%H-%M")
        history['time'].append(dt_string)

    try:
        if best_model_path is not None:
            wandb.save(best_model_path)
    except:
        print('Something went wrong when saving best model')

    print('=' * 100, end='\n\n')

    return history


def train_regression_model(model: nn.Module, criterion, optimizer, train_dataloader, valid_dataloader=None,
                           num_epochs=3, name='model', log_interval=10):
    """
        Trains NN regression on datasets for deblurring and super image resolution
        :param model: The NN model that you would like to train must be of type nn.Module
        :param criterion: The loss function that you would like to use
        :param optimizer: The optimizer that will optimize the NN
        :param train_dataloader: Dataloader that loads training data from dataset
        :param valid_dataloader: Dataloader that loads validation data from dataset
        :param num_epochs: Number of times you want to train the data over
        :param name: name of model that you will use
        :return: A dictionary containing training and validation loss for each epoch
        and list of the kernels after each epoch if the model is a deconv layer
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = criterion.to(device)
    # scheduler = ReduceLROnPlateau(optimizer, 'min')

    metric = PSNR()

    history = {'train_loss': [], 'valid_loss': [], 'psnr': [], 'valid_psnr': []}
    best_model_path = None

    start_time = datetime.now().strftime("%m-%d_%H-%M")
    experiment_name = f'{start_time}_{name}'

    try:
        wandb.watch(model, criterion, log="all", log_freq=10)
    except:
        print('Something went wrong with wandb')

    for epoch in range(num_epochs):
        model.train()

        running_loss = 0.0
        running_psnr = 0.0

        interval_loss = 0.0
        interval_psnr = 0.0

        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))

        for batch_idx, (X, y) in pbar:
            X = X.to(device)
            y = y.to(device)

            with torch.set_grad_enabled(True):
                outputs = model(X)
                loss = criterion(outputs, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                running_psnr += metric(outputs, y).item()
                interval_psnr += metric(outputs, y).item()

            running_loss += loss.item()
            interval_loss += loss.item()

            if log_interval != 0 and (batch_idx + 1) % log_interval == 0:
                avg_train_loss = interval_loss / log_interval
                avg_train_psnr = interval_psnr / log_interval

                pbar.set_postfix({"train_batch_loss": avg_train_loss, "train_batch_psnr": avg_train_psnr})

                try:
                    wandb.log({"train_batch_loss": avg_train_loss, "train_batch_psnr": avg_train_psnr},
                              step=epoch * len(train_dataloader) + batch_idx)
                except:
                    print('Something went wrong with wandb')

                # Reset running loss and PSNR for next log interval
                interval_loss = 0.0
                interval_psnr = 0.0

        history['train_loss'].append(running_loss / len(train_dataloader))
        history['psnr'].append(running_psnr / len(train_dataloader))

        try:
            wandb.log({"train_epoch_loss": running_loss / len(train_dataloader)}, step=epoch)
            wandb.log({"psnr": running_psnr / len(train_dataloader)}, step=epoch)
        except:
            print('Something went wrong with wandb')

        print(f'Epoch {epoch + 1:04d}')
        print(f'train loss: {running_loss / len(train_dataloader):.5f}')
        print(f'train psnr: {history["psnr"][-1]:.5f}')

        if valid_dataloader:
            model.eval()
            valid_running_loss = 0.0
            valid_running_psnr = 0.0
            for X, y in valid_dataloader:
                X = X.to(device)
                y = y.to(device)

                with torch.no_grad():
                    outputs = model(X)
                    loss = criterion(outputs, y)
                    valid_running_psnr += metric(outputs, y).item()

                valid_running_loss += loss.item()

            try:
                wandb.log({"valid_epoch_loss": valid_running_loss / len(valid_dataloader)}, step=epoch)
                wandb.log({"valid_psnr": valid_running_psnr / len(valid_dataloader)}, step=epoch)
            except:
                print('Something went wrong with wandb')

            history['valid_loss'].append(valid_running_loss / len(valid_dataloader))
            print(f'valid loss: {valid_running_loss / len(valid_dataloader):.5f}')
            # scheduler.step(history['valid_loss'][-1])
            history['valid_psnr'].append(valid_running_psnr / len(valid_dataloader))
            print(f'valid psnr: {history["valid_psnr"][-1]:.5f}')

        print('-' * 100)

        if epoch == 0 or min(history['train_loss'][:-1]) > history['train_loss'][-1]:
            best_model_path = save_model(model, experiment_name)

    try:
        wandb.log({"best_train_loss": min(history['train_loss'])})
        wandb.log({"best_valid_loss": min(history['valid_loss'])})
        wandb.log({"best_train_psnr": max(history['train_loss'])})
        wandb.log({"best_valid_psnr": max(history['valid_loss'])})
    except:
        print('Something went wrong when saving best model or best losses')

    try:
        if best_model_path:
            wandb.save(best_model_path)
    except:
        print('Error when saving model to wandb')

    return history


def save_model(model, name, folder='saved_models'):
    if not os.path.exists(folder):
        os.makedirs(folder)

    file_name = f'{folder}/{name}.pt'

    torch.save(model.state_dict(), file_name)
    print(f'Model saved at {file_name}')

    return file_name


def compute_impulse_diffs(dataloader, epoch, model):
    image, _ = next(iter(dataloader))
    try:
        mag, phase = impulse_response_of_model(model, image.size())
        diffs = check_filter_diff(mag)

        mag_images = save_tensor_images(mag)
        phase_images = save_tensor_images(phase)
        diffs_images = save_tensor_images(diffs)
        w_images = save_tensor_images(model.layer1.w)

        wandb.log({f"impulse_response_mag": [wandb.Image(image) for image in mag_images]}, step=epoch)
        wandb.log({f"impulse_response_phase": [wandb.Image(image) for image in phase_images]}, step=epoch)
        wandb.log({f"impulse_response_diffs": [wandb.Image(image) for image in diffs_images]}, step=epoch)
        wandb.log({f"w_images": [wandb.Image(image) for image in w_images]}, step=epoch)

    except:
        print('First layer is not deconv. Not logging impulse responses')
