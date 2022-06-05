import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

database_path = 'C:\\Users\\Rashaad\\Documents\\Postgrad\\Datasets\\dtd\\images'
output_path = 'C:\\Users\\Rashaad\\Documents\\Postgrad\\Datasets\\dtd_grayscale\\images'

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

# Plot first pic of each folder
# for i in range(len(img_disp)):
#     plt.subplot(2, 5, i+1)
#     imgplot = plt.imshow(img_disp[i], cmap='gray')
# plt.show()
# imgs_grayscale = tf.image.rgb_to_grayscale(imgs)
