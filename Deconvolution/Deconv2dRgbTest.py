import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import DeconvMultiDatasetTest as useful
from DeconvDft2dRgbLayer import DeconvDft2dRgbLayer
from DeconvDft2dLayer import DeconvDft2dLayer


def deconv_rgb_forward_pass_test():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    i = random.randint(1, 100)
    xm = tf.concat([[x_train[i]], [x_train[i]], [x_train[i]]], axis=0)
    xm = tf.transpose(xm, perm=[1, 2, 0])

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(xm)
    ax[1].imshow(x_train[i], cmap='gray')
    plt.show()


def mnist_deconvrgb_test():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = DeconvDft2dRgbLayer((1, 3, 3))(inputs)
    x = layers.Flatten()(x)
    x = layers.Dense(512)(x)
    x = layers.Dense(256)(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    results = useful.train_and_evaluate(model, x_train, x_test, y_train, y_test, 20)
    useful.save_data(model.layers[1].w.numpy(), 'deconvrgb_mnist')

    return results


def deconvrgb_mnist_response(w):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype("float32") / 255.0

    i = random.randint(1, 100)
    xm = tf.reshape(x_train[i], (1, 28, 28, 1))
    xm = tf.image.grayscale_to_rgb(xm)

    xm_ = tf.reshape(x_train[i], (1, 28, 28))

    deconv = DeconvDft2dLayer((3, 3))
    deconv.w = w
    print(w)

    deconvrgb = DeconvDft2dRgbLayer((3, 3, 3))
    w = tf.concat([w, w, w], 0)
    deconvrgb.w = w

    ym = deconvrgb(xm)
    ym_ = deconv(xm_)

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(xm[0])
    ax[1].imshow(ym[0], cmap='gray')
    plt.show()


def dtd_test():
    path = 'C:/Users/Rashaad/Documents/Postgrad/Datasets/dtd_copy/images_resized/'
    ds_train, ds_test = get_local_dataset(path)

    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = DeconvDft2dRgbLayer((3, 3, 3))(inputs)
    x = layers.Flatten()(x)
    x = layers.Dense(1024)(x)
    x = layers.Dense(512)(x)
    outputs = layers.Dense(10, activation='relu')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.summary()
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                  metrics=['accuracy'])

    history = model.fit(ds_train, batch_size=64, epochs=10, verbose=2)
    results = model.evaluate(ds_test, batch_size=64, verbose=2)

    print(model.layers[1].w)
    return history


def get_local_dataset(path, split=0.2, seed=123):
    image_size = (32, 32)
    ds_train = tf.keras.preprocessing.image_dataset_from_directory(
        path,
        labels='inferred',
        label_mode='int',  # categorical binary
        color_mode='rgb',
        batch_size=64,
        image_size=image_size,
        shuffle=True,
        seed=123,
        validation_split=0.2,
        subset='training'
    )
    ds_test = tf.keras.preprocessing.image_dataset_from_directory(
        path,
        labels='inferred',
        label_mode='int',  # categorical binary
        color_mode='rgb',
        batch_size=64,
        image_size=image_size,
        shuffle=True,
        seed=123,
        validation_split=0.2,
        subset='validation'
    )
    return ds_train, ds_test


if __name__ == '__main__':
    physical_devices = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    mnist_deconvrgb_test()
    w = useful.load_data('deconvrgb_mnist')
    deconvrgb_mnist_response(w)
