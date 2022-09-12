import os
import random

import tensorflow as tf
from keras import layers
from Convolution.CausalConvLayer import CausalConvLayer

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


def alot_autoregression_test(ds_path, target_size=[256, 256]):
    x_train, y_train = get_dataset(ds_path, target_size)

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

    history = model.fit(x_train, y_train, epochs=20, verbose=2)

    ar_filter = model.layers[0].w

    return history, ar_filter


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


def plot_ar(ar_filter, history, ds_path, target_size=[256, 256], save=''):
    new_shape = None

    x_train, y_train = get_dataset(ds_path, target_size)

    i = random.randint(0, 100)

    new_shape = [1, target_size[0], target_size[1], 1]
    print(new_shape)

    x = tf.reshape(x_train[i], new_shape)

    ar_model = CausalConvLayer((3, 3))
    ar_model.w = ar_filter

    y = ar_model(x)
    y = y * 255

    x = x * 255

    ax = plt.subplot(2, 2, 1)
    plt.imshow(x[0].numpy().astype('uint8'), cmap='gray')
    plt.title('Original image')
    plt.axis('off')

    ax = plt.subplot(2, 2, 2)
    plt.imshow(y[0].numpy().astype('uint8'), cmap='gray')
    plt.title('After autoregression')
    plt.axis('off')

    ax = plt.subplot(2, 1, 2)
    plt.plot(history['mse'])
    plt.title('MSE over epochs')
    plt.ylabel('MSE')
    plt.xlabel('Epochs')

    if save != '':
        plt.savefig(save + '.png')
        plt.clf()
    else:
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
    target_size = [256, 256]
    for i in range(10):
        print('==================================================')
        print()
        print('ITERATION ' + str(i))
        print()
        print('==================================================')

        r = str(random.randint(1, 249))
        ds_path = 'C:/Users/Rashaad/Documents/Postgrad/Datasets/ALOT/alot_grey_quarter/alot_grey4/grey4/' + r
        history = None

        # ds_path = 'C:/Users/Rashaad/Documents/Postgrad/PAR Resources/shared/csir2014/intersection_birds_nama'

        history, ar_filter = alot_autoregression_test(ds_path, target_size)
        useful.save_data(ar_filter.numpy(), 'ar_w_' + r)

        ar_filter = useful.load_data('ar_w_' + r)
        save_filename = 'C:/Users/Rashaad/Documents/Postgrad/Results/Autoregression/alot_' + str(r)
        plot_ar(ar_filter, history.history, ds_path, target_size, save_filename)
