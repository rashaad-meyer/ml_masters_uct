import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

# To Avoid GPU errors
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def convolution_2d(inputs, filter):
    return inputs


# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train = x_train.reshape(-1, 28 * 28).astype("float32")/255.0
# x_test = x_test.reshape(-1, 28 * 28).astype("float32")/255.0
# print(x_train.shape)
#
# A = tf.constant([[[3, 0, 1, 2, 7, 4],
#                   [1, 5, 8, 9, 3, 1],
#                   [2, 7, 2, 5, 1, 3],
#                   [0, 1, 3, 1, 7, 8],
#                   [4, 2, 1, 6, 2, 8],
#                   [2, 4, 5, 2, 3, 9]]], dtype=tf.float32)
# B = tf.concat([A, A], 0)
#
# f = tf.constant([[1, 0, -1],
#                  [1, 0, -1],
#                  [1, 0, -1]], dtype=tf.float32)
#
# out = convolution_2d(B, f)

# toeplitz diagonal
input_shape = 6
kernel_shape = 3
output_shape = input_shape - kernel_shape + 1
kernel = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

row = tf.TensorArray(tf.float32, size=input_shape * input_shape, dynamic_size=False, clear_after_read=False)
zeros = tf.TensorArray(tf.float32, size=input_shape * input_shape, dynamic_size=False, clear_after_read=False)

for i in range(input_shape * input_shape):
    if i % input_shape < kernel_shape and i // input_shape < kernel_shape:
        row = row.write(i, kernel[i // input_shape][(i % input_shape) % kernel_shape])
    else:
        row = row.write(i, 0)
    if i == 0:
        zeros = zeros.write(i, kernel[0][0])
    else:
        zeros = zeros.write(i, 0)

row = row.stack()
zeros = zeros.stack()

out = tf.linalg.LinearOperatorToeplitz(zeros, row).to_dense()

temp = []
for i in range(input_shape ** 2):
    j = 0
    if not (i % input_shape > input_shape - kernel_shape):
        temp.append(out[i])

    if len(temp) >= output_shape ** 2:
        break

out2 = tf.concat([temp], 0)

tf.print("out2")
tf.print(out2, summarize=36)
