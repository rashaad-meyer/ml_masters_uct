import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator


img_height = 384
img_width = 512

model = keras.Sequential([
    layers.Input((img_height, img_width, 1)),
    layers.Conv2D(16, 3, padding='same'),
    layers.Conv2D(32, 3, padding='same'),
    layers.MaxPool2D,
    layers.Flatten(),
    layers.Dense()
])
