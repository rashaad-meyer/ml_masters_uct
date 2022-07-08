import tensorflow as tf
import random
import DeconvMultiDatasetTest as useful
from tensorflow.keras import layers
from Deconvolution.CustomLayers.DeconvDft2dLayer import DeconvDft2dLayer
from Deconvolution.CustomLayers.DeconvDft2dRgbLayer import DeconvDft2dRgbLayer
import matplotlib.pyplot as plt


def get_grayscale_alot_ds(image_size, seed=100, validation_split=0.2):
    ds_train = tf.keras.preprocessing.image_dataset_from_directory(
        'C:/Users/rasha/Documents/MastersUCT/Dataset/ALOT/grey',
        labels='inferred',
        label_mode='int',  # categorical binary
        color_mode='grayscale',
        batch_size=32,
        image_size=image_size[:-1],
        shuffle=True,
        seed=seed,
        validation_split=validation_split,
        subset='training'
    )
    ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
        'C:/Users/rasha/Documents/MastersUCT/Dataset/ALOT/grey',
        labels='inferred',
        label_mode='int',  # categorical binary
        color_mode='grayscale',
        batch_size=32,
        image_size=image_size[:-1],
        shuffle=True,
        seed=seed,
        validation_split=validation_split,
        subset='validation'
    )
    return ds_train, ds_validation


def alot_deconv_test(img_shape):
    (ds_train, ds_validation) = get_grayscale_alot_ds(img_shape)

    def augment(image_label, seed):
        image, label = image_label
        image = tf.image.central_crop(image, 0.7)
        img_size = image.shape[-2]
        image = tf.image.stateless_random_crop(image, size=[img_size, img_size, 1], seed=seed)
        return image, label

    ds_train = (
        ds_train
        .shuffle(1000)
        .map(augment, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    ds_validation = (
        ds_validation
        .map(resize_and_rescale, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    model = tf.keras.Sequential(
        [
            layers.Input((256, 256, 1)),
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


def alot_deconvrgb_test(img_shape):
    (ds_train, ds_validation) = get_grayscale_alot_ds(img_shape)

    model = tf.keras.Sequential(
        [
            layers.Input(img_shape),
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

    history = model.fit(ds_train, epochs=5, verbose=2)
    results = model.evaluate(ds_validation)

    return history, results, model.layers[0].w


def alot_conv_test(img_shape):
    (ds_train, ds_validation) = get_grayscale_alot_ds(img_shape)

    model = tf.keras.Sequential(
        [
            layers.Input(img_shape),
            # layers.Rand
            layers.Conv2D(1, 3),
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

    return history, results, model.layers[0].kernel


def alot_test_comparison(img_shape):
    """
    Train deconv, conv, dense NN on MNIST datasets and return results tuple
    :return: dictionary containing results tuple from each NN
    """
    results = {'deconv': alot_deconvrgb_test(img_shape),
               'conv': alot_conv_test(img_shape)}
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


def alot_deconv_conv_response(c_k, d_k, img_shape):
    i = random.randint(0, 10000)
    (ds_train, ds_validation) = get_grayscale_alot_ds(img_shape, seed=100)

    class_names = ds_train.class_names

    deconv = DeconvDft2dRgbLayer((1, 3, 3))
    conv = layers.Conv2D(1, 5)

    deconv.w = d_k
    conv.kernel = c_k

    plt.figure(figsize=(10, 10))
    for images, labels in ds_train.take(1):
        for i in range(3):
            img = tf.reshape(images[i], (1, 512, 512, 1))
            deconv_out = deconv(img)
            conv_out = conv(img)

            ax = plt.subplot(3, 3, i * 3 + 1)
            plt.imshow(images[i].numpy().astype("uint8"), cmap='gray')
            plt.title(class_names[labels[i]])
            plt.axis("off")

            ax = plt.subplot(3, 3, i * 3 + 2)
            plt.imshow(deconv_out[0].numpy().astype("uint8"), cmap='gray')
            plt.title('Deconv 3x3')
            plt.axis("off")

            ax = plt.subplot(3, 3, i * 3 + 3)
            plt.imshow(conv_out[0].numpy().astype("uint8"), cmap='gray')
            plt.title('Conv 3x3')
            plt.axis("off")

    plt.show()


if __name__ == '__main__':
    img_shape = (600, 600, 1)
    results = alot_deconvrgb_test(img_shape)
