import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from Deconvolution.CustomLayers.DeconvDft2dLayer import DeconvDft2dLayer


# download dataset from this link https://www.robots.ox.ac.uk/~vgg/data/dtd/index.html

def deconv_dtd_test(batch_size=64, epochs=10, lr=0.1, validation_split=0.4, seed=100, plot=False):
    img_height = 231
    img_width = 271
    batch_size = batch_size

    # Initialise Deconvultional NN
    model = tf.keras.Sequential([
        layers.InputLayer((img_height, img_width)),
        DeconvDft2dLayer((3, 3)),
        # layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(47)
    ])

    model = tf.keras.Sequential([
        layers.Input((img_height, img_width, 1)),
        layers.Conv2D(1, 3, padding='same'),
        layers.Flatten(),
        layers.Dense(47)
    ])

    model.summary()

    # Import training data from local directory
    ds_train = tf.keras.preprocessing.image_dataset_from_directory(
        'C:/Users/Rashaad/Documents/Postgrad/Datasets/dtd_grayscale/images/',
        labels='inferred',
        label_mode='int',  # categorical binary
        color_mode='grayscale',
        batch_size=batch_size,
        image_size=(img_height, img_width),
        shuffle=True,
        seed=seed,
        validation_split=validation_split,
        subset='training'
    )

    # Import training data from local directory
    ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
        'C:/Users/Rashaad/Documents/Postgrad/Datasets/dtd_grayscale/images/',
        labels='inferred',
        label_mode='int',  # categorical binary
        color_mode='grayscale',
        batch_size=batch_size,
        image_size=(img_height, img_width),
        shuffle=True,
        seed=seed,
        validation_split=validation_split,
        subset='validation'
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=[tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), ],
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

    history = model.fit(ds_train,
                        epochs=epochs,
                        validation_data=ds_validation,
                        validation_steps=1)

    print("Evaluate")
    deconv_results = model.evaluate(ds_validation, batch_size=2*batch_size)

    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Accuracy and Loss plots')
        ax1.plot(history.history['sparse_categorical_accuracy'])
        ax1.set(xlabel='Epochs', ylabel='Accuracy')
        ax2.plot(history.history['loss'])
        ax2.set(xlabel='Epochs', ylabel='Loss')
        # ax2.set_ylim([0, 5000])
        plt.show()


if __name__ == '__main__':
    physical_devices = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    batch_size = 100
    epochs = 20
    lr = 0.001
    validation_split = 0.1
    deconv_dtd_test(batch_size=10, epochs=epochs, lr=lr, validation_split=validation_split, plot=True)
