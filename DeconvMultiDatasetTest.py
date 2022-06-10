import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import cifar10

from DeconvDft2dLayer import DeconvDft2dLayer


def mnist_dataset_test():

    print('\n=================================================================')
    print('=================================================================')
    print('                       Deconvolutional Model                     ')
    print('=================================================================')
    print('=================================================================\n')

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Functional API for Deconv Layer
    inputs = tf.keras.Input(shape=(28, 28))
    x = DeconvDft2dLayer((3, 3))(inputs)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu", name="second_layer")(x)
    outputs = layers.Dense(10, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.summary()

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=2)
    model.evaluate(x_test, y_test, batch_size=32, verbose=2)

    print('\n=================================================================')
    print('=================================================================')
    print('                         Dense Model                             ')
    print('=================================================================')
    print('=================================================================\n')

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28 * 28).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28 * 28).astype("float32") / 255.0

    # Functional API for Dense layer
    inputs = tf.keras.Input(shape=(28 * 28))
    x = layers.Dense(512, activation="relu", name="first_layer")(inputs)
    x = layers.Dense(256, activation="relu", name="second_layer")(x)
    outputs = layers.Dense(10, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.summary()

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=2)
    model.evaluate(x_test, y_test, batch_size=32, verbose=2)


def cifar10_dataset_test():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    def my_model():
        inputs = tf.keras.Input(shape=(32, 32, 3))
        x = layers.Conv2D(32, 3)(inputs)
        x = layers.BatchNormalization()(x)
        x = tf.keras.activations.relu(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(64, 3)(x)
        x = layers.BatchNormalization()(x)
        x = tf.keras.activations.relu(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(128, 3)(x)
        x = layers.BatchNormalization()(x)
        x = tf.keras.activations.relu(x)
        x = layers.Flatten()(x)
        outputs = layers.Dense(10, activation='relu')(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

    def my_model_deconv():
        inputs = tf.keras.Inpput(shape=(32, 32, 3))

    model = my_model()
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        metrics=['accuracy'],
    )

    model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=2)
    print("Evaluate")
    model.evaluate(x_test, y_test, batch_size=64, verbose=2)


if __name__ == '__main__':
    physical_devices = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # cifar10_dataset_test()
    mnist_dataset_test()
