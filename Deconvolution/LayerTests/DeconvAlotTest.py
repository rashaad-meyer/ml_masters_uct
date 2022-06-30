import tensorflow as tf
import random
import DeconvMultiDatasetTest as useful
from tensorflow.keras import layers
from Deconvolution.CustomLayers.DeconvDft2dLayer import DeconvDft2dLayer


def get_grayscale_alot_ds(seed=100, validation_split=0.1):
    image_size = (256, 256)
    ds_train = tf.keras.preprocessing.image_dataset_from_directory(
        'C:/Users/Rashaad/Documents/Postgrad/Datasets/alot_grey/grey/',
        labels='inferred',
        label_mode='int',  # categorical binary
        color_mode='grayscale',
        batch_size=100,
        image_size=image_size,
        shuffle=True,
        seed=123,
        validation_split=0.2,
        subset='training'
    )
    ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
        'C:/Users/Rashaad/Documents/Postgrad/Datasets/alot_grey/grey/',
        labels='inferred',
        label_mode='int',  # categorical binary
        color_mode='grayscale',
        batch_size=100,
        image_size=image_size,
        shuffle=True,
        seed=123,
        validation_split=0.2,
        subset='validation'
    )
    return ds_train, ds_validation


def alot_deconv_test():
    (ds_train, ds_validation) = get_grayscale_alot_ds()

    # def augment(x, y):
    #     seed = (random.randint(1, 1000), random.randint(1, 1000))
    #     image = tf.image.stateless_random_crop(x, size=(-1, 256, 256, 1), seed=seed)
    #     return image, y
    # ds_train = ds_train.map(augment)

    model = tf.keras.Sequential(
        [
            layers.Input((256, 256, 1)),
            layers.Conv2D(1, 3, padding='same'),
            layers.Conv2D(1, 3, padding='same'),
            layers.MaxPooling2D(),
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
