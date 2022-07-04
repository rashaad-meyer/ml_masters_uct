import tensorflow as tf
import random
import DeconvMultiDatasetTest as useful
from tensorflow.keras import layers
from Deconvolution.CustomLayers.DeconvDft2dLayer import DeconvDft2dLayer
from Deconvolution.CustomLayers.DeconvDft2dRgbLayer import DeconvDft2dRgbLayer


def get_grayscale_alot_ds(image_size, seed=100, validation_split=0.2):
    ds_train = tf.keras.preprocessing.image_dataset_from_directory(
        'C:/Users/Rashaad/Documents/Postgrad/Datasets/ALOT/alot_grey_quarter/alot_grey4/grey4',
        labels='inferred',
        label_mode='int',  # categorical binary
        color_mode='grayscale',
        batch_size=32,
        image_size=image_size,
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
        image_size=image_size,
        shuffle=True,
        seed=seed,
        validation_split=validation_split,
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

    history = model.fit(ds_train, epochs=10, verbose=2)
    results = model.evaluate(ds_validation)

    return history, results


def alot_deconvrgb_test():
    img_shape = (256, 256)
    (ds_train, ds_validation) = get_grayscale_alot_ds(img_shape)
    print(tf.__version__)

    model = tf.keras.Sequential(
        [
            layers.Input((256, 256, 1)),
            # layers.Rand
            DeconvDft2dRgbLayer((1, 3, 3)),
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


def alot_conv_test():
    img_shape = (256, 256)
    (ds_train, ds_validation) = get_grayscale_alot_ds(img_shape)
    print(tf.__version__)

    model = tf.keras.Sequential(
        [
            layers.Input((256, 256, 1)),
            # layers.Rand
            layers.Conv2D(1, (3, 3)),
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


if __name__ == '__main__':
    alot_deconvrgb_test()
    alot_deconv_test()
