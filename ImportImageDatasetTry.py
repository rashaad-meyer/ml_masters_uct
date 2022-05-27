import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# download dataset from this link https://people.csail.mit.edu/celiu/CVPR2010/FMD/

img_height = 384
img_width = 512
batch_size = 100

model = tf.keras.Sequential([
    layers.Input((img_height, img_width, 1)),
    layers.Conv2D(16, 3, padding='same'),
    layers.Conv2D(32, 3, padding='same'),
    layers.MaxPooling2D(),
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
    seed=123,
    validation_split=0.1,
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
    seed=123,
    validation_split=0.1,
    subset='validation'
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=[tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), ],
    metrics=['accuracy']
)

model.fit(ds_train, epochs=10, verbose=2)
model.evaluate(ds_validation, batch_size=10, verbose=2)

# plot first image
# img = mpimg.imread(ds_train.file_paths[0])
# img_gray = tf.image.rgb_to_grayscale(img)
# print(img_gray)
# imgplot = plt.imshow(img_gray, cmap='gray')
# plt.show()
