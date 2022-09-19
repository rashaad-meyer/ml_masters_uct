import random

import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from keras import layers
from matplotlib import gridspec

from Deconvolution.CustomLayers.DeconvDft2dLayer import DeconvDft2dLayer as Deconv2D
from Deconvolution.LayerTests import DeconvMultiDatasetTest as useful
import numpy as np


def deconv_super_res_test(save_w=False):
    (x_train, y_train), (x_test, y_test) = get_img_super_res_ds()

    inputs = keras.Input(shape=(256, 256, 1))
    outputs = Deconv2D((3, 3))(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(),
    )

    w0 = model.layers[1].w.numpy()
    history = model.fit(x_train, y_train, batch_size=32, epochs=20, verbose=2)
    model.evaluate(x_test, y_test, batch_size=32, verbose=2)
    wf = model.layers[1].w.numpy()

    if save_w:
        useful.save_data(wf, 'super_res_deconv_wf')
        useful.save_data(w0, 'super_res_deconv_w0')

    return history


def deconv_gauss_blur_test(ds_path, save_w=False):
    print("Loading training data....")
    x_train = get_imgs_from_dir(ds_path + '/train/gauss_blur/')
    y_train = get_imgs_from_dir(ds_path + '/train/high_res/')

    # Define Functional model
    inputs = keras.Input(shape=(256, 256, 1))
    outputs = Deconv2D((3, 3))(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(),
    )
    w0 = model.layers[1].w.numpy()
    model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=2)
    wf = model.layers[1].w.numpy()

    if save_w:
        useful.save_data(wf, 'gauss_blur_deconv_wf')
        useful.save_data(w0, 'gauss_blur_deconv_w0')

    return


def get_img_super_res_ds():
    x_train = useful.load_data('super_res_x_train')
    y_train = useful.load_data('super_res_y_train')
    x_test = useful.load_data('super_res_x_test')
    y_test = useful.load_data('super_res_y_test')
    return (x_train, y_train), (x_test, y_test)


def save_data_in_csv():
    base_path = 'C:/Users/Rashaad/Documents/Postgrad/Datasets/Super resolution/' + \
                'Image Super Resolution Aditya/dataset'

    x_train = imgs_to_tensor(base_path + '/train/low_res')
    useful.save_data(x_train.numpy(), 'super_res_x_train')
    print('x_train saved')

    y_train = imgs_to_tensor(base_path + '/train/high_res')
    useful.save_data(y_train.numpy(), 'super_res_y_train')
    print('y_train saved')

    x_test = imgs_to_tensor(base_path + '/val/low_res')
    useful.save_data(x_test.numpy(), 'super_res_x_test')
    print('x_test saved')

    y_test = imgs_to_tensor(base_path + '/val/high_res')
    useful.save_data(y_test.numpy(), 'super_res_y_test')
    print('y_test saved')


def imgs_to_tensor(data_path):
    # Get input data

    img_names = os.listdir(data_path)
    x_train = []

    for img_name in img_names:
        img_path = os.path.join(data_path, img_name)
        img = tf.keras.utils.load_img(img_path,
                                      color_mode='grayscale',
                                      target_size=(256, 256))
        img = tf.keras.preprocessing.image.img_to_array(img)
        x_train.append(img)

    x_train = tf.constant(x_train)

    x_train = x_train / 255.0

    return x_train


def plot_super_res_test_results(history=None):
    i = random.randint(0, 140)

    # load data from tests
    x_test = useful.load_data('super_res_x_test')
    y_test = useful.load_data('super_res_y_test')
    wf = useful.load_data('super_res_deconv_wf')
    w0 = useful.load_data('super_res_deconv_w0')

    x = x_test[i:i + 2]
    y = y_test[i:i + 2]

    deconv = Deconv2D((3, 3))

    deconv.w = wf
    y_predf = deconv(x)

    deconv.w = w0
    y_pred0 = deconv(x)

    pad_w = tf.constant([[0, 0], [1, 0]])
    wf_ = tf.pad(wf, pad_w, mode='CONSTANT', constant_values=1)
    wf_ = tf.reshape(wf_, (3, 3))
    w0_ = tf.pad(w0, pad_w, mode='CONSTANT', constant_values=1)
    w0_ = tf.reshape(w0_, (3, 3))

    plt.figure(figsize=(10, 10))

    ax = plt.subplot(2, 2, 1)
    plt.imshow(x[0], cmap='gray')
    ax.set_title('Low res')
    plt.axis('off')

    ax = plt.subplot(2, 2, 2)
    plt.imshow(y[0], cmap='gray')
    ax.set_title('High Res')
    plt.axis('off')

    ax = plt.subplot(2, 2, 3)
    plt.imshow(y_pred0[0].numpy(), cmap='gray')
    ax.set_title('Predicted w0')
    plt.axis('off')

    ax = plt.subplot(2, 2, 4)
    plt.imshow(y_predf[0].numpy(), cmap='gray')
    ax.set_title('Predicted wf')
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
    plt.plot(history.history['loss'])
    ax.set_title('MSE over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')

    plt.show()


def plot_gauss_blur_test_results():
    i = random.randint(0, 140)

    # load data from tests
    base_path = 'C:/Users/Rashaad/Documents/Postgrad/Datasets/Super resolution/' + \
                'Image Super Resolution Aditya/dataset'
    x_train = get_imgs_from_dir(base_path + '/train/gauss_blur/')
    y_train = get_imgs_from_dir(base_path + '/train/high_res/')
    wf = useful.load_data('gauss_blur_deconv_wf')
    w0 = useful.load_data('gauss_blur_deconv_w0')

    x = x_train[i:i + 2]
    y = y_train[i:i + 2]

    deconv = Deconv2D((3, 3))

    deconv.w = wf
    y_predf = deconv(x)

    deconv.w = w0
    y_pred0 = deconv(x)

    pad_w = tf.constant([[0, 0], [1, 0]])
    wf_ = tf.pad(wf, pad_w, mode='CONSTANT', constant_values=1)
    wf_ = tf.reshape(wf_, (3, 3))
    w0_ = tf.pad(w0, pad_w, mode='CONSTANT', constant_values=1)
    w0_ = tf.reshape(w0_, (3, 3))

    plt.figure(figsize=(10, 10))

    ax = plt.subplot(2, 4, 1)
    plt.imshow(y[0], cmap='gray')
    ax.set_title('High Res')
    plt.axis('off')

    ax = plt.subplot(2, 4, 2)
    plt.imshow(y_predf[0].numpy(), cmap='gray')
    ax.set_title('Predicted wf')
    plt.axis('off')

    ax = plt.subplot(2, 4, 3)
    plt.imshow(y_pred0[0].numpy(), cmap='gray')
    ax.set_title('Predicted w0')
    plt.axis('off')

    ax = plt.subplot(2, 4, 4)
    plt.imshow(x[0], cmap='gray')
    ax.set_title('Gauss blur')
    plt.axis('off')

    ax = plt.subplot(2, 2, 3)
    ax.table(wf_.numpy(), loc='center')
    ax.set_title('kernel after training')
    ax.axis('off')

    ax = plt.subplot(2, 2, 4)
    ax.table(w0_.numpy(), loc='center')
    ax.set_title('kernel before training')
    ax.axis('off')

    plt.show()


def gauss_blur_ds(ds_path):
    # get data that needs to be transformed
    imgs = get_imgs_from_dir(ds_path)

    kernel = np.array([[1 / 16, 1 / 8, 1 / 16],
                       [1 / 8, 1 / 4, 1 / 8],
                       [1 / 16, 1 / 8, 1 / 16]])

    convfn = layers.Conv2D(1, (3, 3), padding='same')
    convfn.kernel = kernel
    y = convfn(imgs)

    img_names = os.listdir(ds_path)
    base_path = ds_path + '/../gauss_blur/'

    if not os.path.exists(base_path):
        os.makedirs(base_path)

    for i in range(y.shape[0]):
        if img_names[i][-4:] == '.png':
            img_path = base_path + img_names[i]
            tf.keras.utils.save_img(img_path, y[i])
            print('Saved image @ ' + img_path)


def get_imgs_from_dir(ds_path, target_size=[256, 256]):
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

    return x_train


if __name__ == '__main__':
    print(tf.__version__)
    base_path = 'C:/Users/Rashaad/Documents/Postgrad/Datasets/Super resolution/' + \
                'Image Super Resolution Aditya/dataset'
    # deconv_gauss_blur_test(base_path, save_w=True)
    # plot_gauss_blur_test_results()
    history = deconv_super_res_test(save_w=True)
    plot_super_res_test_results(history)
    # save_data_in_csv()
