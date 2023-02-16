import os

import tensorflow as tf
import random

from keras_preprocessing.image import ImageDataGenerator

from keras import layers
from TensorFlow.Deconvolution.CustomLayers.DeconvDft2dLayer import DeconvDft2dLayer
from TensorFlow.Deconvolution.CustomLayers.DeconvDft2dRgbLayer import DeconvDft2dRgbLayer
import matplotlib.pyplot as plt
from TensorFlow.Augmentation.RandomCropLayer import RandomCrop
import DeconvMultiDatasetTest as useful

import pandas as pd


def get_grayscale_alot_ds(image_size, seed=100, validation_split=0.2):
    ds_train = tf.keras.preprocessing.image_dataset_from_directory(
        'C:/Users/Rashaad/Documents/Postgrad/Datasets/ALOT/alot_grey_quarter/alot_grey4/grey4',
        labels='inferred',
        label_mode='int',  # categorical binary
        color_mode='grayscale',
        batch_size=32,
        image_size=image_size[:-1],
        shuffle=True,
        seed=seed,
        validation_split=validation_split,
        subset='training'
    )
    ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
        'C:/Users/Rashaad/Documents/Postgrad/Datasets/ALOT/alot_grey_quarter/alot_grey4/grey4',
        labels='inferred',
        label_mode='int',  # categorical binary
        color_mode='grayscale',
        batch_size=32,
        image_size=image_size[:-1],
        shuffle=True,
        seed=seed,
        validation_split=validation_split,
        subset='validation'
    )
    return ds_train, ds_validation


def alot_deconv_test(img_shape):
    (ds_train, ds_validation) = get_grayscale_alot_ds(img_shape)

    model = tf.keras.Sequential(
        [
            layers.Input(img_shape),
            DeconvDft2dLayer((3, 3)),
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


def alot_deconv_with_augmentation_test(img_shape):
    aug_shape = [460, 460, 1]
    # aug_shape = list(img_shape)

    ds_train = tf.keras.preprocessing.image_dataset_from_directory(
        'C:/Users/Rashaad/Documents/Postgrad/Datasets/ALOT/alot_grey_quarter/alot_grey4/grey4',
        labels='inferred',
        label_mode='int',  # categorical binary
        color_mode='grayscale',
        batch_size=32,
        image_size=img_shape[:-1],
        shuffle=True,
        seed=123,
        validation_split=0.2,
        subset='training'
    )
    ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
        'C:/Users/Rashaad/Documents/Postgrad/Datasets/ALOT/alot_grey_quarter/alot_grey4/grey4',
        labels='inferred',
        label_mode='int',  # categorical binary
        color_mode='grayscale',
        batch_size=32,
        image_size=aug_shape[:-1],
        shuffle=True,
        seed=123,
        validation_split=0.2,
        subset='validation'
    )

    def augment(img, label):
        aug_shape1 = aug_shape
        img = random_central_crop(img, aug_shape1)
        return img, label

    ds_train = ds_train.map(augment)

    model = tf.keras.Sequential(
        [
            layers.Input(aug_shape),
            DeconvDft2dLayer((3, 3)),
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


def alot_deconv_test(img_shape):
    (ds_train, ds_validation) = get_grayscale_alot_ds(img_shape)

    model = tf.keras.Sequential(
        [
            layers.Input(img_shape),
            # layers.Rand
            DeconvDft2dLayer((3, 3)),
            layers.Flatten(),
            layers.Dense(250),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=[tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)],
        metrics=["accuracy"],
    )

    w_i = model.layers[0].w.numpy()

    history = model.fit(ds_train, epochs=10, verbose=2)
    results = model.evaluate(ds_validation)

    w_f = model.layers[0].w.numpy()

    print('W before:')
    print(w_i)
    print('===========================')
    print('W after:')
    print(w_f)

    return history, results, w_i, w_f


def alot_deconv_test_with_augmentation(img_shape):
    (ds_train, ds_validation) = get_grayscale_alot_ds(img_shape)
    crop_size = (256, 256, 1)

    model = tf.keras.Sequential(
        [
            layers.Input(img_shape),
            # layers.Rand
            DeconvDft2dLayer((3, 3)),
            layers.Flatten(),
            layers.Dense(250),
        ]
    )

    def augment(img_label, seed):
        label = img_label[1]
        img = random_central_crop(img_label[0], crop_size)
        return img, label

    ds_train.map(augment)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=[tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)],
        metrics=["accuracy"],
    )

    history = model.fit(ds_train, epochs=20, verbose=2)
    results = model.evaluate(ds_validation)

    return history, results, model.layers[0].w


def alot_conv_test(img_shape):
    (ds_train, ds_validation) = get_grayscale_alot_ds(img_shape)

    model = tf.keras.Sequential(
        [
            layers.Input(img_shape),
            # layers.Rand
            layers.Conv2D(1, 3),
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

    return history, results, model.layers[0].kernel


def alot_test_comparison(img_shape):
    """
    Train deconv, conv, dense NN on MNIST datasets and return results tuple
    :return: dictionary containing results tuple from each NN
    """
    results = {'deconv': alot_deconv_test(img_shape),
               'conv': alot_conv_test(img_shape)}
    return results


def plot_results(trains_results):
    """
    Plots loss and accuracy from mnist_data_comparison()
    :param trains_results: Results from mnist_test_comparison()
    :return:
    """
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Accuracy and Loss plots')

    for i in trains_results.keys():
        ax1.plot(trains_results[i].history['accuracy'])
        ax2.plot(trains_results[i].history['loss'])

    ax1.set(xlabel='Epochs', ylabel='Accuracy')
    ax1.legend(trains_results.keys(), loc='lower right')
    ax2.set(xlabel='Epochs', ylabel='Loss')
    ax2.legend(trains_results.keys(), loc='upper right')

    plt.show()


def alot_deconv_conv_response(c_k, d_k, img_shape):
    i = random.randint(0, 10000)
    (ds_train, ds_validation) = get_grayscale_alot_ds(img_shape, seed=i)

    class_names = ds_train.class_names

    deconv = DeconvDft2dRgbLayer((1, 3, 3))
    conv = layers.Conv2D(1, 5)

    deconv.w = d_k
    conv.kernel = c_k

    img_reshape = list(img_shape)
    img_reshape.insert(0, 1)

    plt.figure(figsize=(10, 10))
    for images, labels in ds_train.take(1):
        for i in range(3):
            img = tf.reshape(images[i], tuple(img_reshape))
            deconv_out = deconv(img)
            conv_out = conv(img)

            ax = plt.subplot(3, 3, i * 3 + 1)
            plt.imshow(images[i].numpy().astype("uint8"), cmap='gray')
            plt.title(class_names[labels[i]])
            plt.axis("off")

            ax = plt.subplot(3, 3, i * 3 + 2)
            plt.imshow(deconv_out[0].numpy().astype("uint8"), cmap='gray')
            plt.title('Deconv 3x3')
            plt.axis("off")

            ax = plt.subplot(3, 3, i * 3 + 3)
            plt.imshow(conv_out[0].numpy().astype("uint8"), cmap='gray')
            plt.title('Conv 3x3')
            plt.axis("off")

    plt.show()


def random_central_crop(img, crop_size):
    y0 = int((img.shape[-3] - crop_size[-3]) / 2)
    x0 = int((img.shape[-2] - crop_size[-2]) / 2)
    rand_xy = tf.random.uniform((2,), minval=-20, maxval=20, dtype=tf.int32)
    img = tf.image.crop_to_bounding_box(img, y0 + rand_xy[0], x0 + rand_xy[1], crop_size[-3], crop_size[-2])
    return img.numpy()


def random_crop_test(img_shape):
    i = random.randint(0, 10000)
    (ds_train, ds_validation) = get_grayscale_alot_ds(img_shape, seed=100)

    class_names = ds_train.class_names

    img_shape = list(img_shape).insert(0, 1)

    plt.figure(figsize=(10, 10))
    for images, labels in ds_train.take(1):
        for i in range(3):
            rcrop_out1 = random_central_crop(images[i], (100, 100, 1))
            rcrop_out2 = random_central_crop(images[i], (100, 100, 1))

            ax = plt.subplot(3, 3, i * 3 + 1)
            plt.imshow(images[i].numpy().astype("uint8"), cmap='gray')
            plt.title(class_names[labels[i]])
            plt.axis("off")

            ax = plt.subplot(3, 3, i * 3 + 2)
            plt.imshow(rcrop_out1.numpy().astype("uint8"), cmap='gray')
            plt.title('Deconv 3x3')
            plt.axis("off")

            ax = plt.subplot(3, 3, i * 3 + 3)
            plt.imshow(rcrop_out2.numpy().astype("uint8"), cmap='gray')
            plt.title('Conv 3x3')
            plt.axis("off")

    plt.show()


def deconv_alot_impulse_response(c_k, d_k):
    """
    Plots an image frequency response of the output of a convolutional and deconvolutional layer
    when 2D impulse response is used as an input
    :param c_k: Convolutional kernel
    :param d_k: Deconvolutional kernel
    :return:
    """
    # Prepare impulse response with right dimension
    xmd = tf.pad([[[1.0]]], [[0, 0], [0, 127], [0, 256]])
    xmd = tf.reshape(xmd, (1, xmd.shape[-2], xmd.shape[-1], 1))

    xmc = tf.pad([[[1.0]]], [[0, 0], [0, 20], [0, 127]])
    xmc = tf.reshape(xmc, (1, xmc.shape[-2], xmc.shape[-1], 1))
    # Initialise layers
    convfn = layers.Conv2D(1, (3, 3), padding='same')
    deconvfn = DeconvDft2dLayer((3, 3))

    # Set layer filters
    convfn.kernel = c_k
    deconvfn.w = d_k

    # Calcualte response
    ymc = convfn(xmc)
    ymd = deconvfn(xmd)

    # Reshape output, so it can be plotted
    # ymc = tf.reshape(ymc, (ymc.shape[-3], ymc.shape[-2]))
    # ymd = tf.reshape(ymd, (ymd.shape[-2], ymd.shape[-1]))

    ymcf = tf.math.abs(tf.signal.rfft2d(ymc))
    ymdf = tf.math.abs(tf.signal.rfft2d(ymd))

    ax = plt.subplot(2, 1, 1)
    plt.imshow(ymcf[0], cmap='gray')
    plt.title('Conv 3x3')

    ax = plt.subplot(2, 1, 2)
    plt.imshow(ymdf[0], cmap='gray')
    plt.title('Deconv 3x3')

    plt.show()


def save_history_csv(w_i, w_f, history, test_acc, input_shape, aug=False):
    train_hist = {'w_i': [w_i], 'w_f': [w_f], 'history': [history], 'test_acc': [test_acc], 'data_aug': [aug]}
    df = pd.DataFrame(train_hist)
    output_path = 'csv_files/alot_test_results.csv'
    df.to_csv(output_path, mode='a', header=not os.path.exists(output_path))


if __name__ == '__main__':
    img_shape = (256, 256, 1)
    print(tf.__version__)

    for i in range(5):
        print('====================================')
        print('              Loop ' + str(i))
        print('====================================')
        history, results, w_i, w_f = alot_deconv_test(img_shape)
        save_history_csv(w_i, w_f, history.history['accuracy'], img_shape, results)

    # useful.save_data(d_k, 'alot_deconv_w')
    # c_k = useful.load_data('alot_conv_kernel')
    # d_k = useful.load_data('alot_deconv_w')
    # alot_deconv_conv_response(c_k, d_k, img_shape)
