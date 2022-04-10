import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from CustomConvolutionLayer import CustomConvolutionLayer

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)


# Test if forward function works
def test_forward_fcn():
    conv_2d = CustomConvolutionLayer(3, [6, 6])

    inputs = tf.constant([[3, 0, 1, 2, 7, 4,
                           1, 5, 8, 9, 3, 1,
                           2, 7, 2, 5, 1, 3,
                           0, 1, 3, 1, 7, 8,
                           4, 2, 1, 6, 2, 8,
                           2, 4, 5, 2, 3, 9]], dtype=tf.float32)

    out = conv_2d.call(inputs)

    tf.print(tf.reshape(out, [4, 4]))


def test_with_nn():
    print("Testing with NN")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28 * 28).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28 * 28).astype("float32") / 255.0

    # Sequential API (Very Convenient, not very flexible)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(28 * 28)),
            CustomConvolutionLayer(3, [28, 28]),
            layers.Dense(512, activation="relu"),
            layers.Dense(256, activation="relu"),
            layers.Dense(10)
        ]
    )
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=["accuracy"],
    )
    model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=2)
    model.evaluate(x_test, y_test, batch_size=32, verbose=2)


def test_with_single_layer_nn():
    print("Testing with single layer NN")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28 * 28).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28 * 28).astype("float32") / 255.0

    # Sequential API (Very Convenient, not very flexible)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(28 * 28)),
            CustomConvolutionLayer(3, [28, 28])
        ]
    )
    tf.print('--- self.w before training ---')
    tf.print(model.layers[0].w, summarize=28)

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=["accuracy"],
    )
    model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=2)
    model.evaluate(x_test, y_test, batch_size=32, verbose=2)

    tf.print('--- self.w after training ---')
    tf.print(model.layers[0].w, summarize=28)


if __name__ == '__main__':
    test_with_single_layer_nn()
    # test_with_nn()
