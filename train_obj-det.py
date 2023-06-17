"""
Main file for training Yolo model on Pascal VOC dataset

"""
import argparse

import torch
import torchvision.transforms as transforms
import torch.optim as optim
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader

from PyTorch.Models.CnnModules import TwoLayerCNN
from PyTorch.ObjectDetection.model import Yolov1
from PyTorch.ObjectDetection.dataset import VOCDataset
from PyTorch.ObjectDetection.utils import (
    non_max_suppression,
    mean_average_precision,
    intersection_over_union,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint,
)
from PyTorch.ObjectDetection.loss import YoloLoss
from PyTorch.util.helper_functions import get_voc_ds
from PyTorch.util.training_functions import compute_impulse_diffs
from train_classification import read_json_objects

seed = 123
torch.manual_seed(seed)

IMG_SIZE = (448, 448)
BASE_DIR = "data/obj-det"
IMG_DIR = f"{BASE_DIR}/images"
LABEL_DIR = f"{BASE_DIR}/labels"


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes


transform = Compose([transforms.Resize(IMG_SIZE), transforms.ToTensor(), ])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds_len", default=-1, type=int, help="Length of dataset")
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size for training")
    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs for training")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="Learning rate")
    parser.add_argument("--weight_decay", default=0, type=float, help="Weight decay for the optimizer")
    parser.add_argument('--multi', dest='multi', action='store_true')

    args = parser.parse_args()
    return args


def train_fn(train_loader, model, optimizer, loss_fn):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(DEVICE)

    loop = tqdm(train_loader, leave=True)
    losses = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss=loss.item())

    mean_loss = sum(losses) / len(losses)
    print(f"Mean loss was {mean_loss}")

    return mean_loss


def main():
    args = parse_args()
    get_voc_ds(BASE_DIR)
    print('Loading model...')

    ds_length = None
    if args.ds_len != -1:
        ds_length = args.ds_len

    print('Loading train dataset...')
    train_dataset = VOCDataset(
        "data/obj-det/train.csv",
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
        ds_length=ds_length
    )

    print('Loading test dataset...')
    test_dataset = VOCDataset(
        "data/obj-det/test.csv", transform=transform, img_dir=IMG_DIR, label_dir=LABEL_DIR,
    )

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True)

    if args.multi:
        wandb.login()

        configs = read_json_objects('experiment_csv/obj-det.txt')

        for hyperparams in configs:
            with wandb.init(project="YoloV1", config=hyperparams):
                config = wandb.config

                S, B, C = 7, 2, 20
                num_classes = S * S * (C + B * 5)

                model = TwoLayerCNN(**config, num_classes=num_classes, img_size=IMG_SIZE)
                optimizer = optim.Adam(
                    model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
                )
                loss_fn = YoloLoss()

                print('Starting training...')
                train_yolo(args, loss_fn, model, optimizer, train_loader)

    else:
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
        optimizer = optim.Adam(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        loss_fn = YoloLoss()

        print('Starting training...')
        train_yolo(args, loss_fn, model, optimizer, train_loader)


def train_yolo(args, loss_fn, model, optimizer, train_loader):
    wandb.watch(model, loss_fn, log="all", log_freq=10)
    compute_impulse_diffs(train_loader, 0, model)

    for epoch in range(args.epochs):
        print(f'Epoch {epoch + 1: 5d}/{args.epochs}:')
        loss = train_fn(train_loader, model, optimizer, loss_fn)

        print('Computing mAP...')
        pred_boxes, target_boxes = get_bboxes(
            train_loader, model, iou_threshold=0.5, threshold=0.4
        )
        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )
        wandb.log({"epoch_loss": loss, "mAP": mean_avg_prec}, step=epoch)
        compute_impulse_diffs(train_loader, epoch + 1, model)

        print(f"Train mAP: {mean_avg_prec}")
        print('==================================================================\n')


if __name__ == "__main__":
    main()
