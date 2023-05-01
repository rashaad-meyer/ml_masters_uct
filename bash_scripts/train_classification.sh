#!/bin/bash

cd "$(dirname "$0")/.." || exit

python train_classification.py --num_epochs 20 --deconv

python train_classification.py --num_epochs 20