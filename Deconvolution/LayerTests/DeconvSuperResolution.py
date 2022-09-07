import random

import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from keras import layers
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

    model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=2)
    model.evaluate(x_test, y_test, batch_size=32, verbose=2)
    w = model.layers[1].w

    if save_w:
        useful.save_data(w, 'super_res_deconv_w')

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


def plot_super_res_test_results():
    i = random.randint(0, 140)

    # load data from tests
    x_test = useful.load_data('super_res_x_test')
    y_test = useful.load_data('super_res_y_test')
    w = useful.load_data('super_res_deconv_w')

    x = x_test[i:i + 2]
    y = y_test[i:i + 2]

    deconv = Deconv2D((3, 3))
    y_pred = deconv(x)

    plt.figure(figsize=(10, 10))

    for i in range(x.shape[0]):
        ax = plt.subplot(2, 3, i * 3 + 1)
        plt.imshow(y[i], cmap='gray')
        ax.set_title('High Res')
        plt.axis('off')

        ax = plt.subplot(2, 3, i * 3 + 2)
        plt.imshow(y_pred[i].numpy(), cmap='gray')
        ax.set_title('Predicted')
        plt.axis('off')

        ax = plt.subplot(2, 3, i * 3 + 3)
        plt.imshow(x[i], cmap='gray')
        ax.set_title('Low Res')
        plt.axis('off')

    plt.show()


if __name__ == '__main__':
    print(tf.__version__)
    save_data_in_csv()
    deconv_super_res_test(save_w=True)
    plot_super_res_test_results()
    # save_data_in_csv()
