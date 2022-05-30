import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from DeconvDft2dLayer import DeconvDft2dLayer

# download dataset from this link https://people.csail.mit.edu/celiu/CVPR2010/FMD/

img_height = 384
img_width = 512
batch_size = 10
input_shape = (img_height, img_width)

model = tf.keras.Sequential()


# model = tf.keras.Sequential([
#     layers.Input((img_height, img_width, 1)),
#     layers.Conv2D(16, 3, padding='same'),
#     layers.Conv2D(32, 3, padding='same'),
#     layers.MaxPooling2D(),
#     layers.Flatten(),
#     layers.Dense(10)
# ])

model = tf.keras.Sequential([
    layers.Input((img_height, img_width)),
    DeconvDft2dLayer((3, 3)),
    # layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(10)
])

#                  METHOD 1                  #
# ========================================== #
#       Using dataset_from_directory         #
# ========================================== #
ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    'C:/Users/Rashaad/Documents/Postgrad/Datasets/FlickrMaterialDatabase_grayscale/image/',
    labels='inferred',
    label_mode='int',  # categorical binary
    color_mode='grayscale',
    batch_size=batch_size,
    image_size=(img_height, img_width),
    shuffle=True,
    seed=100,
    validation_split=0.4,
    subset='training'
)

ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
    'C:/Users/Rashaad/Documents/Postgrad/Datasets/FlickrMaterialDatabase_grayscale/image/',
    labels='inferred',
    label_mode='int',  # categorical binary
    color_mode='grayscale',
    batch_size=batch_size,
    image_size=(img_height, img_width),
    shuffle=True,
    seed=100,
    validation_split=0.4,
    subset='validation'
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss=[tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), ],
    metrics=['accuracy']
)

model.fit(ds_train, epochs=20, verbose=2)
model.evaluate(ds_validation, batch_size=64, verbose=2)