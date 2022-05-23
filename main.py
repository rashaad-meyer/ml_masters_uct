import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import time

# To Avoid GPU errors. Turn off if you don't have a GPU
# physical_devices = tf.config.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(physical_devices[0], True)


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

# populating 2d tensor
def create_toeplitz():
    input_shape = [3, 3]
    kernel_size = 2

    kernel = tf.reshape(tf.range(1, kernel_size ** 2 + 1, dtype=tf.float32), [kernel_size, kernel_size])

    t0 = time.time()
    input_dim = tf.cast(input_shape[-1], tf.int32)
    input_size = input_dim * input_dim

    output_dim = input_dim - kernel_size + 1
    output_size = output_dim * output_dim

    row = tf.TensorArray(tf.float32, size=input_size, dynamic_size=False, clear_after_read=False)
    zeros = tf.TensorArray(tf.float32, size=input_size, dynamic_size=False, clear_after_read=False)

    t1 = time.time()
    for i in range(input_size):
        if i % input_dim < kernel_size and i // input_dim < kernel_size:
            row = row.write(i, kernel[i // input_dim][(i % input_dim) % kernel_size])
        else:
            row = row.write(i, 0)
        if i == 0:
            zeros = zeros.write(i, kernel[0][0])
        else:
            zeros = zeros.write(i, 0)
    zeros = tf.zeros([input_size])
    t2 = time.time()

    row = row.stack()
    # zeros = zeros.stack()

    toeplitz = tf.linalg.LinearOperatorToeplitz(zeros, row).to_dense()

    t3 = time.time()
    # Take a slices of toeplitz matrix because toeplitz is not exactly the same as the matrix that we need
    temp = []
    for i in range(input_size):
        if not (i % input_dim > input_dim - kernel_size):
            temp.append(toeplitz[i])

        if len(temp) >= output_size:
            break

    t4 = time.time()

    weight_matrix = tf.concat([temp], 0)

    t_f = time.time()

    t0_ = t_f - t0
    t1_ = t1 - t0
    t2_ = t2 - t1
    t3_ = t3 - t2
    t4_ = t4 - t3

    print("t0 = %s seconds " % t0_)
    print("t4 = %s seconds " % t4_)
    print("t3 = %s seconds " % t3_)
    print("t2 = %s seconds " % t2_)
    print("t1 = %s seconds " % t1_)

    return weight_matrix


def create_conv_mat_indices(input_dim, kernel_dim):
    kernel_size = kernel_dim ** 2

    output_dim = input_dim - kernel_dim + 1
    output_size = output_dim ** 2

    # calculate index displacements for each row
    j = 0
    k_i = []
    for i in range(kernel_size):
        if i % kernel_dim == 0 and i != 0:
            j = j + (input_dim - kernel_dim)

        k_i.append(j)
        j = j + 1

    indices = []
    j = 0

    # calculate indices for entire matrix
    for i in range(output_size):

        if i % output_dim == 0 and i != 0:
            j = j + input_dim - output_dim

        for v in k_i:
            indices.append([i, j + v])

        j = j + 1
    return indices


def create_conv_mat(input_shape, kernel):
    # Calculate dimensions
    input_dim = input_shape[-1]
    input_size = input_dim ** 2

    kernel_dim = kernel.shape[-1]
    kernel_size = kernel_dim ** 2

    output_dim = input_dim - kernel_dim + 1
    output_size = output_dim ** 2

    indices = create_conv_mat_indices(input_shape[-1], kernel_dim)

    # flatten kernel then repeat it for output_size amount of times
    flat_kernel = tf.reshape(kernel, kernel_size)
    kernel_tile = tf.tile(flat_kernel, [output_size])

    st = tf.SparseTensor(indices=indices, values=kernel_tile, dense_shape=[output_size, input_size])
    st1 = tf.sparse.to_dense(st)
    return st1


if __name__ == '__main__':
    print()
