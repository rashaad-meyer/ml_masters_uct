import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)


class CustomConvolutionLayer(layers.Layer):
    def __init__(self, kernel_size):
        super(CustomConvolutionLayer, self).__init__()
        self.kernel_size = kernel_size
        self.w = self.add_weight(
            name="w",
            shape=(kernel_size, kernel_size),
            dtype=tf.float32,
            initializer="random_normal",
            trainable=True,
        )

    def call(self, inputs):
        # do padding
        pad = int((self.kernel_size - 1)/2)
        padding = tf.constant([[0, 0], [pad, pad], [pad, pad]])
        inputs = tf.pad(inputs, padding, "CONSTANT")

        input_shape = inputs.shape
        tf.print(input_shape)
        batch_size = input_shape[0]

        output_shape = input_shape[1] - self.kernel_size + 1

        # iterate through data
        output_temp = []
        for c in range(batch_size):
            # iterate through each pixel/data point
            temp = []
            for i in range(output_shape):
                row = tf.TensorArray(tf.float32, size=output_shape, dynamic_size=False, clear_after_read=False)
                for j in range(output_shape):
                    element = tf.constant(0, dtype=tf.float32)
                    for u in range(self.kernel_size):
                        for v in range(self.kernel_size):
                            n = tf.multiply(self.w[u][v], inputs[c][i + u][j + v])
                            element = tf.add(element, n)

                    row = row.write(j, element)
                row = row.stack()
                temp.append(row)

            output_temp.append(tf.concat([temp], 0))
        output = tf.concat([output_temp], 0)
        return output


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Sequential API (Very Convenient, not very flexible)
model = tf.keras.Sequential(
    [
        tf.keras.layers.InputLayer(input_shape=(28, 28),
                                   batch_size=1),
        CustomConvolutionLayer(3),
        CustomConvolutionLayer(3),
        layers.Flatten(),
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

model.fit(x_train, y_train, batch_size=32, epochs=1, verbose=2)
model.evaluate(x_test, y_test, batch_size=32, verbose=2)
