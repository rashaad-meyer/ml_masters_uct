import os

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.datasets import mnist

from CustomGradientTest import CustomLayer


def custom_dense_test():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28 * 28).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28 * 28).astype("float32") / 255.0
    # Sequential API (Very convenient, not very flexible)

    model = keras.Sequential(
        [
            keras.Input(shape=(28 * 28)),
            CustomLayer(512),
            layers.Activation(keras.activations.relu),
            CustomLayer(256),
            layers.Activation(keras.activations.relu),
            CustomLayer(10),
            layers.Activation(keras.activations.softmax),
        ]
    )

    # train custom dense layer model
    print('\n================================')
    print('\tCustom Dense Layer')
    print('================================')

    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=["accuracy"],
    )

    model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=2)
    model.evaluate(x_test, y_test, batch_size=32, verbose=2)


def tf_dense_test():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28 * 28).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28 * 28).astype("float32") / 255.0

    model = keras.Sequential(
        [
            keras.Input(shape=(28 * 28)),
            layers.Dense(512, activation="relu"),
            layers.Dense(256, activation="relu"),
            layers.Dense(10),
            layers.Activation(keras.activations.softmax),
        ]
    )

    # train tensorflow dense layer model
    print('================================')
    print('\tTF Dense Layer')
    print('================================')

    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=["accuracy"],
    )

    model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=2)
    model.evaluate(x_test, y_test, batch_size=32, verbose=2)


if __name__ == '__main__':
    custom_dense_test()
    tf_dense_test()
