import tensorflow as tf
from tensorflow.keras import layers


class CustomConvolutionLayer(layers.Layer):

    def __init__(self, kernel_size, input_shape):
        super(CustomConvolutionLayer, self).__init__()
        self.kernel_size = kernel_size
        self.kernel = self.add_weight(
            name="kernel",
            shape=(kernel_size, kernel_size),
            dtype=tf.float32,
            initializer="random_normal",
            trainable=True,
        )
        self.forward_conv_mat = self.create_conv_mat(input_shape, kernel_size)

    @tf.custom_gradient
    def custom_op(self, inputs):
        result = tf.transpose(tf.matmul(self.w, inputs, transpose_b=True))

        def custom_grad(dy, variable):
            return dy

        return result, custom_grad

    def call(self, inputs):
        return tf.transpose(tf.matmul(self.w, inputs, transpose_b=True))

    @staticmethod
    def create_conv_mat(input_shape, kernel):
        # Calculate dimensions
        input_dim = input_shape[-1]
        input_size = input_dim ** 2

        kernel_dim = kernel.shape[-1]
        kernel_size = kernel_dim ** 2

        output_dim = input_dim - kernel_dim + 1
        output_size = output_dim ** 2

        j = 0
        k_i = []
        for i in range(kernel_size):
            if i % kernel_dim == 0 and i != 0:
                j = j + (input_dim - kernel_dim)

            k_i.append(j)
            j = j + 1

        indices = []
        j = 0

        for i in range(output_size):

            if i % output_dim == 0 and i != 0:
                j = j + input_dim - output_dim

            for v in k_i:
                indices.append([i, j + v])

            j = j + 1

        flat_kernel = tf.reshape(kernel, kernel_size)
        kernel_tile = tf.tile(flat_kernel, [output_size])

        st = tf.SparseTensor(indices=indices, values=kernel_tile, dense_shape=[output_size, input_size])
        st1 = tf.sparse.to_dense(st)
        return st1
