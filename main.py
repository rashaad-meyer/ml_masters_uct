import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

# To Avoid GPU errors
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def convolution_2d(inputs, filter):
    input_shape = inputs.shape
    batch_size = input_shape[0]
    filter_size = filter.shape[0]

    output_shape = input_shape[1] - filter_size + 1

    # iterate through data
    output_temp = []
    for c in range(batch_size):
        # iterate through each pixel/data point
        temp = []
        for i in range(output_shape):
            row = tf.TensorArray(tf.float32, size=output_shape, dynamic_size=False, clear_after_read=False)
            for j in range(output_shape):
                element = tf.constant(0, dtype=tf.float32)
                for u in range(filter_size):
                    for v in range(filter_size):
                        n = tf.multiply(filter[u][v], inputs[c][i + u][j + v])
                        element = tf.add(element, n)

                row = row.write(j, element)
            row = row.stack()
            temp.append(row)

        output_temp.append(tf.concat([temp], 0))
    output = tf.concat([output_temp], 0)
    return output


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28 * 28).astype("float32")/255.0
x_test = x_test.reshape(-1, 28 * 28).astype("float32")/255.0
print(x_train.shape)

A = tf.constant([[[3, 0, 1, 2, 7, 4],
                  [1, 5, 8, 9, 3, 1],
                  [2, 7, 2, 5, 1, 3],
                  [0, 1, 3, 1, 7, 8],
                  [4, 2, 1, 6, 2, 8],
                  [2, 4, 5, 2, 3, 9]]], dtype=tf.float32)
B = tf.concat([A, A], 0)

f = tf.constant([[1, 0, -1],
                 [1, 0, -1],
                 [1, 0, -1]], dtype=tf.float32)

out = convolution_2d(B, f)
