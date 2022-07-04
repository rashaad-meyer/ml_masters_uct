import tensorflow as tf
import random
import DeconvMultiDatasetTest as useful
from tensorflow.keras import layers
from Deconvolution.CustomLayers.DeconvDft2dLayer import DeconvDft2dLayer


def get_grayscale_alot_ds(image_size, seed=100, validation_split=0.1):
    ds_train = tf.keras.preprocessing.image_dataset_from_directory(
        'C:/Users/Rashaad/Documents/Postgrad/Datasets/ALOT/alot_grey_quarter/alot_grey4/grey4',
        labels='inferred',
        label_mode='int',  # categorical binary
        color_mode='grayscale',
        batch_size=32,
        image_size=image_size,
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
        image_size=image_size,
        shuffle=True,
        seed=123,
        validation_split=0.2,
        subset='validation'
    )
    return ds_train, ds_validation


def alot_deconv_test():
    img_shape = (256, 256)
    (ds_train, ds_validation) = get_grayscale_alot_ds(img_shape)
    print(tf.__version__)

    model = tf.keras.Sequential(
        [
            layers.Input((256, 256)),
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

    model.fit(ds_train, epochs=10, verbose=2)
    model.evaluate(ds_validation)


if __name__ == '__main__':
    alot_deconv_test()
