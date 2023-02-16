import os
import random

import tensorflow as tf
from keras import layers
from TensorFlow.Convolution.CausalConvLayer import CausalConvLayer

import matplotlib.pyplot as plt
import numpy as np

import DeconvMultiDatasetTest as useful


def conv_rms_test():
    # input
    # causal conv like deconv
    # centre extract
    # flatten? maybe not needed check if you can do without
    # make mean square loss function
    ds_train = tf.keras.preprocessing.image_dataset_from_directory(
        'C:/Users/Rashaad/Documents/Postgrad/Datasets/ALOT/alot_grey_quarter/alot_grey4/grey4',
        labels='inferred',
        label_mode='int',  # categorical binary
        color_mode='grayscale',
        batch_size=32,
        image_size=(256, 256),
        shuffle=True,
        seed=100,
        validation_split=0.2,
        subset='training'
    )
    ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
        'C:/Users/Rashaad/Documents/Postgrad/Datasets/ALOT/alot_grey_quarter/alot_grey4/grey4',
        labels='inferred',
        label_mode='int',  # categorical binary
        color_mode='grayscale',
        batch_size=32,
        image_size=(256, 256),
        shuffle=True,
        seed=100,
        validation_split=0.2,
        subset='validation'
    )

    model = tf.keras.Sequential(
        [
            layers.Input((256, 256, 1)),
            CausalConvLayer((3, 3), 1, 1),
            layers.Flatten(),
            layers.Dense(250),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=[tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)],
        metrics=["accuracy"],
    )

    history = model.fit(ds_train, epochs=10, verbose=2)
    results = model.evaluate(ds_validation)

    return history, results


def custom_loss(y_pred):
    return tf.math.reduce_mean(tf.square(y_pred))


def custom_training_loop():
    data_path = 'C:/Users/Rashaad/Documents/Postgrad/Datasets/ALOT/alot_grey_quarter/alot_grey4/grey4/1/'
    img_names = os.listdir(data_path)
    imgs = []
    for img_name in img_names:
        img_path = os.path.join(data_path, img_name)
        img = tf.keras.utils.load_img(img_path,
                                      color_mode='grayscale',
                                      target_size=(300, 300))
        img = tf.keras.preprocessing.image.img_to_array(img)
        imgs.append(img)

    imgs = tf.constant(imgs)

    x_train = imgs / 255.0
    y_train = tf.zeros(x_train.shape)

    model = tf.keras.Sequential(
        [
            layers.Input((300, 300, 1)),
            CausalConvLayer((3, 3)),
            # layers.Conv2D(1, 3)
        ]
    )

    epochs = 2
    optimizer = tf.keras.optimizers.Adam()
    loss_metric = tf.keras.metrics.MeanSquaredError()

    x_train = np.split(x_train, 5)

    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        for x_batch in x_train:
            with tf.GradientTape() as tape:
                logits = model(x_batch, training=True)
                loss = custom_loss(logits)

            gradients = tape.gradients(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))

        print('Loss over epoch:')
        print(loss)

    return imgs


def alot_autoregression_test(ds_name, target_size=[256, 256]):
    x_train, y_train = load_ds_from_csv(ds_name)

    print(x_train.shape)

    model = tf.keras.Sequential(
        [
            layers.Input((256, 256, 1)),
            CausalConvLayer((3, 3)),
            # layers.Conv2D(1, 3, padding='same')
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.SGD(),
        loss=[tf.keras.losses.MeanSquaredError()],
        metrics='mse'
    )
    w0 = model.layers[0].w.numpy()

    history = model.fit(x_train, y_train, epochs=20, verbose=2)

    wf = model.layers[0].w.numpy()

    useful.save_data(w0, ds_name + '_w0')
    useful.save_data(wf, ds_name + '_wf')

    return history, w0, wf


def get_dataset(ds_path, target_size=[256, 256]):
    img_names = os.listdir(ds_path)
    imgs = []

    for img_name in img_names:
        if img_name[-4:] == '.png':
            img_path = os.path.join(ds_path, img_name)
            img = tf.keras.utils.load_img(img_path,
                                          color_mode='grayscale',
                                          target_size=target_size)
            img = tf.keras.preprocessing.image.img_to_array(img)
            imgs.append(img)
    aug_imgs = imgs

    imgs_tensor = tf.constant(aug_imgs)
    x_train = imgs_tensor / 255.0
    y_train = tf.zeros(x_train.shape)

    return x_train, y_train


def save_ds_to_csv(x_train, y_train, name):
    """
    :param x_train: input data
    :param y_train: output data
    :param name: <name of dataset>_<class_name>_<algorithm>
    :return:
    """
    useful.save_data(x_train, name + '_x_train')
    useful.save_data(y_train, name + '_y_train')


def load_ds_from_csv(name):
    x_train = useful.load_data(name + '_x_train')
    y_train = useful.load_data(name + '_y_train')
    return x_train, y_train


def plot_ar(history, ds_name, target_size=None):
    if target_size is None:
        target_size = [256, 256]
    new_shape = None

    x_train, y_train = load_ds_from_csv(ds_name)

    i = random.randint(0, 100)

    x = x_train[i:i + 2]

    wf = useful.load_data(ds_name + '_wf')
    w0 = useful.load_data(ds_name + '_w0')

    ar_model = CausalConvLayer((3, 3))
    ar_model.w = wf
    yf = ar_model(x)
    ar_model.w = w0
    y0 = ar_model(x)

    pad_w = tf.constant([[0, 0], [1, 0]])
    wf_ = tf.pad(wf, pad_w, mode='CONSTANT', constant_values=1)
    wf_ = tf.reshape(wf_, (3, 3))
    w0_ = tf.pad(w0, pad_w, mode='CONSTANT', constant_values=1)
    w0_ = tf.reshape(w0_, (3, 3))

    plt.figure(figsize=(10, 10))

    ax = plt.subplot(1, 3, 1)
    plt.imshow(x[0], cmap='gray')
    plt.title('Original image')
    plt.axis('off')

    ax = plt.subplot(1, 3, 2)
    plt.imshow(yf[0].numpy(), cmap='gray')
    plt.title('After autoregressive training')
    plt.axis('off')

    ax = plt.subplot(1, 3, 3)
    plt.imshow(y0[0].numpy(), cmap='gray')
    plt.title('Before autoregressive training')
    plt.axis('off')

    plt.show()

    # plot on differet plot
    plt.figure(figsize=(10, 10))

    ax = plt.subplot(2, 2, 1)
    ax.table(w0_.numpy(), loc='center')
    ax.set_title('kernel before training')
    ax.axis('tight')
    ax.axis('off')

    ax = plt.subplot(2, 2, 2)
    ax.table(wf_.numpy(), loc='center')
    ax.set_title('kernel after training')
    ax.axis('tight')
    ax.axis('off')

    ax = plt.subplot(2, 1, 2)
    plt.plot(history['mse'])
    plt.title('MSE over epochs')
    plt.ylabel('MSE')
    plt.xlabel('Epochs')

    plt.show()


def inverse_autoregression_test():
    # TODO inverse of autoregression then white noise through autregression
    return 0


def random_central_crop(img, crop_size):
    y0 = int((img.shape[-3] - crop_size[-3]) / 2)
    x0 = int((img.shape[-2] - crop_size[-2]) / 2)
    rand_xy = tf.random.uniform((2,), minval=-20, maxval=20, dtype=tf.int32)
    img = tf.image.crop_to_bounding_box(img, y0 + rand_xy[0], x0 + rand_xy[1], crop_size[-3], crop_size[-2])
    return img


if __name__ == '__main__':
    print(tf.__version__)
    # ds_path = 'C:/Users/Rashaad/Documents/Postgrad/Datasets/ALOT/alot_grey_quarter/alot_grey4/grey4/1/'
    ds_name = 'alot_1_autoregression'
    history, w0, wf = alot_autoregression_test(ds_name)
    plot_ar(history.history, ds_name)
