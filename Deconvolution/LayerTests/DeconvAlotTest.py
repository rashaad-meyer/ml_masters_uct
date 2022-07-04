import tensorflow as tf
import random
import DeconvMultiDatasetTest as useful
from tensorflow.keras import layers
from Deconvolution.CustomLayers.DeconvDft2dLayer import DeconvDft2dLayer
from Deconvolution.CustomLayers.DeconvDft2dRgbLayer import DeconvDft2dRgbLayer
import matplotlib.pyplot as plt


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


def alot_test_comparison():
    """
    Train deconv, conv, dense NN on MNIST datasets and return results tuple
    :return: dictionary containing results tuple from each NN
    """
    results = {'deconv': alot_deconvrgb_test(),
               'conv': alot_conv_test()}
    return results


def plot_results(trains_results):
    """
    Plots loss and accuracy from mnist_data_comparison()
    :param trains_results: Results from mnist_test_comparison()
    :return:
    """
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Accuracy and Loss plots')

    for i in trains_results.keys():
        ax1.plot(trains_results[i].history['accuracy'])
        ax2.plot(trains_results[i].history['loss'])

    ax1.set(xlabel='Epochs', ylabel='Accuracy')
    ax1.legend(trains_results.keys(), loc='lower right')
    ax2.set(xlabel='Epochs', ylabel='Loss')
    ax2.legend(trains_results.keys(), loc='upper right')

    plt.show()


if __name__ == '__main__':
    results = alot_test_comparison()
    train_results = {'deconv': results['deconv'][0],
                     'conv': results['conv'][0]}
    plot_results(train_results)
