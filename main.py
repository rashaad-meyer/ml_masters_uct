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

    # TODO can use tf.concat([r1,r2,r3...], 1) to concat array
    # iterate through data
    for c in range(batch_size):
        # iterate through each pixel/data point
        for i in range(input_shape[1] - filter_size + 1):
            for j in range(input_shape[2] - filter_size + 1):
                element = tf.constant(0)
                for u in range(filter_size):
                    for v in range(filter_size):
                        n = tf.multiply(filter[u][v], inputs[c][i+u][j+v])
                        element = tf.add(element, n)

            tf.print()
        tf.print()

    return inputs


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28 * 28).astype("float32")/255.0
x_test = x_test.reshape(-1, 28 * 28).astype("float32")/255.0
print(x_train.shape)

A = tf.constant([[[3, 0, 1, 2, 7, 4],
                  [1, 5, 8, 9, 3, 1],
                  [2, 7, 2, 5, 1, 3],
                  [0, 1, 3, 1, 7, 8],
                  [4, 2, 1, 6, 2, 8],
                  [2, 4, 5, 2, 3, 9]]])
B = tf.concat([A, A], 0)

f = tf.constant([[1, 0, -1],
                 [1, 0, -1],
                 [1, 0, -1]])
# tf.print(B)
# print(B.shape)
# convolution_2d(B, f)

ta = tf.TensorArray(tf.float32, size=(3, 3), dynamic_size=False, clear_after_read=False)
ta = ta.write((0, 0), 10)
ta = ta.write((1, 1), 20)
ta = ta.write((2, 2), 30)
ta = ta.stack()

tb = tf.TensorArray(tf.float32, size=3, dynamic_size=False, clear_after_read=False)
tb = tb.write(0, 10)
tb = tb.write(1, 20)
tb = tb.write(2, 30)
tb = tb.stack()

tf.print(ta)

out = tf.multiply(ta, tb)
tf.print(out)
