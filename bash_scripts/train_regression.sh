#!/bin/bash

cd "$(dirname "$0")/.."

python train_regression.py --multi --ds 91-image --same_size