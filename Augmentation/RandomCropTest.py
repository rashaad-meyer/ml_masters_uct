import random
import matplotlib.pyplot as plt
import tensorflow as tf
from Deconvolution.LayerTests import DeconvAlotTest
from RandomCropLayer import RandomCrop


def random_crop_test(img_shape):
    i = random.randint(0, 10000)
    (ds_train, ds_validation) = DeconvAlotTest.get_grayscale_alot_ds(img_shape, seed=100)

    class_names = ds_train.class_names

    random_crop = RandomCrop((400, 400))

    img_shape = list(img_shape).insert(0, 1)

    plt.figure(figsize=(10, 10))
    for images, labels in ds_train.take(1):
        for i in range(3):
            img = tf.reshape(images[i], (1, 600, 600, 1))
            rcrop_out1 = random_crop(img)
            rcrop_out2 = random_crop(img)

            ax = plt.subplot(3, 3, i * 3 + 1)
            plt.imshow(images[i].numpy().astype("uint8"), cmap='gray')
            plt.title(class_names[labels[i]])
            plt.axis("off")

            ax = plt.subplot(3, 3, i * 3 + 2)
            plt.imshow(rcrop_out1[0].numpy().astype("uint8"), cmap='gray')
            plt.title('Deconv 3x3')
            plt.axis("off")

            ax = plt.subplot(3, 3, i * 3 + 3)
            plt.imshow(rcrop_out2[0].numpy().astype("uint8"), cmap='gray')
            plt.title('Conv 3x3')
            plt.axis("off")

    plt.show()


if __name__ == '__main__':
    print(tf.__version__)
    random_crop_test((600, 600, 1))
