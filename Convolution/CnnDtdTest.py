import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt


# download dataset from this link https://www.robots.ox.ac.uk/~vgg/data/dtd/index.html

def cnn_dtd_test():
    img_height = 231
    img_width = 271
    batch_size = 64

    # Initialise Deconvultional NN
    model = tf.keras.Sequential([
        layers.InputLayer((231, 271, 1)),

        layers.Conv2D(32, (3, 3), padding='same', activation="relu", input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2), strides=2),

        layers.Conv2D(64, (3, 3), padding='same', activation="relu"),
        layers.MaxPooling2D((2, 2), strides=2),

        layers.Flatten(),
        layers.Dense(100, activation="relu"),
        layers.BatchNormalization(),
        layers.Dense(47, activation="softmax")
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
        seed=123,
        validation_split=0.1,
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
        seed=123,
        validation_split=0.1,
        subset='validation'
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=[tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), ],
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

    history = model.fit(ds_train,
                        epochs=200,
                        validation_data=ds_validation,
                        validation_steps=1)

    # Plot accuracy and loss
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Accuracy and Loss plots')
    ax1.plot(history.history['sparse_categorical_accuracy'])
    ax1.plot(history.history['val_sparse_categorical_accuracy'])
    ax1.set(xlabel='Epochs', ylabel='Accuracy')
    ax1.legend(['acc', 'val_acc'], loc='lower right')
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set(xlabel='Epochs', ylabel='Loss')
    ax2.legend(['loss', 'val_loss'], loc='lower right')
    plt.show()


if __name__ == '__main__':
    cnn_dtd_test()
