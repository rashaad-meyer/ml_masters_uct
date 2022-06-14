import os
import tensorflow as tf
from tensorflow import keras
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from tensorflow.python.keras.datasets import cifar10


def convert_directory_to_grayscale(database_path, output_path):
    folders = os.listdir(database_path)
    img_size = [231, 271]
    for i in folders:
        img_base_path = os.path.join(database_path, i)
        img_names = os.listdir(img_base_path)
        path_folder = os.path.join(output_path, i)
        if not os.path.exists(path_folder):
            os.makedirs(path_folder)
        print(i)

        for j in img_names:
            if j[-4:] != '.png' and j[-4:] != '.jpg':
                print(j[-4:])
                continue

            img_path = os.path.join(img_base_path, j)
            img = mpimg.imread(img_path)

            if len(img.shape) == 3:
                img_grayscale = tf.image.rgb_to_grayscale(img)
                path_full = os.path.join(path_folder, j)
                img_grayscale = tf.image.resize(img_grayscale, img_size)
                tf.keras.utils.save_img(path_full, img_grayscale)
            else:
                img = tf.reshape(img, (img.shape[0], img.shape[1], 1))
                path_full = os.path.join(path_folder, j)
                img = tf.image.resize(img, img_size)
                tf.keras.utils.save_img(path_full, img)


def convert_data_to_grayscale(x_train, y_train, labels, output_path, x_test=None, y_test=None):
    # Create directories
    if x_test is None:
        x_test = []
    for i in labels:
        folder_path = os.path.join(output_path, i)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    # Convert images and put them into directory
    img_count = [0] * 10
    for i in range(len(x_train)):
        img = tf.image.rgb_to_grayscale(x_train[i])
        img_name = labels[y_train[i][0]] + '_' + '{0:04}'.format(img_count[y_train[i][0]]) + '.png'
        print(img_name)
        img_count[y_train[i][0]] += 1
        folder_path = os.path.join(output_path, labels[y_train[i][0]])
        img_path = os.path.join(folder_path, img_name)
        tf.keras.utils.save_img(img_path, img)

    if x_test is not None:
        for i in range(len(x_test)):
            img = tf.image.rgb_to_grayscale(x_test[i])
            img_name = labels[y_test[i][0]] + '_' + '{0:04}'.format(img_count[y_test[i][0]]) + '.png'
            print(img_name)
            img_count[y_test[i][0]] += 1
            folder_path = os.path.join(output_path, labels[y_test[i][0]])
            img_path = os.path.join(folder_path, img_name)
            tf.keras.utils.save_img(img_path, img)


if __name__ == '__main__':
    # database_path = 'C:\\Users\\Rashaad\\Documents\\Postgrad\\Datasets\\dtd\\images'
    # output_path = 'C:\\Users\\Rashaad\\Documents\\Postgrad\\Datasets\\dtd_grayscale\\images'
    # convert_directory_to_grayscale(database_path, output_path)

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    output_path = 'C:\\Users\\Rashaad\\Documents\\Postgrad\\Datasets\\cifar10_grayscale\\images'
    convert_data_to_grayscale(x_train, y_train, labels, output_path, x_test, y_test)
