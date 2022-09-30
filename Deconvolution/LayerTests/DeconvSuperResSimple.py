import tensorflow as tf
from tensorflow import keras
import numpy as np
import os


def get_imgs_from_dir(ds_path, target_size=[256, 256]):
    img_names = os.listdir(ds_path)
    imgs = []

    for img_name in img_names:
        if img_name[-4:] == '.png':
            img_path = os.path.join(ds_path, img_name)
            img = tf.keras.utils.load_img(img_path,
                                          color_mode='rgb',
                                          target_size=target_size)
            img = tf.keras.preprocessing.image.img_to_array(img)
            imgs.append(img)
    aug_imgs = imgs

    imgs_tensor = tf.constant(aug_imgs, dtype=tf.float32)
    x_train = imgs_tensor / 255.0

    return x_train


def easy_two_by_two_blur_ds(ds_path):
    # get data that needs to be transformed
    imgs = get_imgs_from_dir(ds_path)

    k = np.array([[0.2, 0.0],
                  [0.0, 0.5]], dtype=np.float32)
    k0 = np.array([[0.0, 0.0],
                   [0.0, 0.0]], dtype=np.float32)

    kr = np.array([k, k0, k0], dtype=np.float32)
    kg = np.array([k0, k, k0], dtype=np.float32)
    kb = np.array([k0, k0, k], dtype=np.float32)

    kf = np.array([kr, kg, kb, kb])
    kf = tf.transpose(kf, perm=[2, 3, 1, 0])

    # y = tf.nn.conv2d(imgs, kf, strides=[1, 1, 1, 1], padding='SAME')

    conv2d = keras.layers.Conv2D(3, (2, 2), padding='SAME')
    conv2d.kernel = kf
    y = conv2d(imgs)

    print(conv2d.kernel)

    img_names = os.listdir(ds_path)
    base_path = ds_path + '/../two_by_two_blur/'

    if not os.path.exists(base_path):
        os.makedirs(base_path)

    for i in range(y.shape[0]):
        if img_names[i][-4:] == '.png':
            img_path = base_path + img_names[i]
            tf.keras.utils.save_img(img_path, y[i])
            print('Saved image @ ' + img_path)


if __name__ == '__main__':
    print(tf.__version__)
    ds_path = 'C:/Users/Rashaad/Documents/Postgrad/Datasets/Super resolution/Image Super Resolution ' \
              'Aditya/dataset/train/high_res'
    easy_two_by_two_blur_ds(ds_path)
    # save_data_in_csv()
