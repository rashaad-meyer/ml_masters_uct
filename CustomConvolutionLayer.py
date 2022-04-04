import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)


class CustomConvolutionLayer(layers.Layer):
    def __init__(self, kernel_size):
        super(CustomConvolutionLayer, self).__init__()
        self.weight_matrix = None
        self.kernel_size = kernel_size
        self.kernel = self.add_weight(
            name="w",
            shape=(kernel_size, kernel_size),
            dtype=tf.float32,
            initializer="random_normal",
            trainable=True,
        )

    def build(self, input_shape):
        # create toeplitz to matmul with
        input_dim = tf.cast(input_shape[-1], tf.int32)
        input_size = input_dim*input_dim

        output_dim = input_dim - self.kernel_size + 1
        output_size = output_dim*output_dim

        row = tf.TensorArray(tf.float32, size=input_size, dynamic_size=False, clear_after_read=False)
        zeros = tf.TensorArray(tf.float32, size=input_size, dynamic_size=False, clear_after_read=False)

        for i in range(input_size):
            if i % input_dim < self.kernel_size and i // input_dim < self.kernel_size:
                row = row.write(i, self.kernel[i // input_dim][(i % input_dim) % self.kernel_size])
            else:
                row = row.write(i, 0)
            if i == 0:
                zeros = zeros.write(i, self.kernel[0][0])
            else:
                zeros = zeros.write(i, 0)

        row = row.stack()
        zeros = zeros.stack()

        toeplitz = tf.linalg.LinearOperatorToeplitz(zeros, row).to_dense()

        # Take a slices of toeplitz matrix because toeplitz is not exactly the same as the matrix that we need
        temp = []
        for i in range(input_size):
            if not (i % input_dim > input_dim - self.kernel_size):
                temp.append(toeplitz[i])

            if len(temp) >= output_size:
                break

        self.weight_matrix = tf.concat([temp], 0)

    def call(self, inputs):
        return tf.matmul(self.kernel_matrix, inputs)


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28 * 28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28 * 28).astype("float32") / 255.0

# Sequential API (Very Convenient, not very flexible)
model = tf.keras.Sequential(
    [
        tf.keras.layers.InputLayer(input_shape=(28*28),
                                   batch_size=1),
        CustomConvolutionLayer(3),
        CustomConvolutionLayer(3),
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
