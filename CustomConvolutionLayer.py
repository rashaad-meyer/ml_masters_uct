import tensorflow as tf
from tensorflow.keras import layers


class CustomConvolutionLayer(layers.Layer):

    def __init__(self, kernel_dim, input_shape):
        super(CustomConvolutionLayer, self).__init__()
        self.kernel_dim = kernel_dim

        self.kernel = self.add_weight(
            name="kernel",
            shape=(kernel_dim, kernel_dim),
            dtype=tf.float32,
            initializer="random_normal",
            trainable=True,
        )
        output_dim = input_shape[-1] - kernel_dim + 1
        self.forward_conv_mat = self.create_conv_mat(input_shape, kernel_dim)
        self.dLdW_indices = self.create_conv_mat_indices(input_shape[-1], output_dim)

    @tf.custom_gradient
    def custom_op(self, inputs):
        result = tf.transpose(tf.matmul(self.forward_conv_mat, inputs, transpose_b=True))

        def custom_grad(dldz, variable):
            kernel_size = self.kernel_dim**2

            dldz_size = dldz[-1]**2

            input_size = inputs.shape[-1]**2

            flat_dldz = tf.reshape(dldz, dldz_size)
            dldz_tile = tf.tile(flat_dldz, [kernel_size])

            st = tf.SparseTensor(indices=self.dLdW_indices, values=dldz_tile, dense_shape=[kernel_size, input_size])
            st = tf.sparse.to_dense(st)

            dldf = tf.reshape(tf.matmul(st, inputs), [self.kernel_dim, self.kernel_dim])

            return dldf

        return result, custom_grad

    def call(self, inputs):
        return self.custom_op(inputs)

    def create_conv_mat(self, input_shape, kernel):
        # Calculate dimensions
        input_dim = input_shape[-1]
        input_size = input_dim ** 2

        kernel_dim = kernel.shape[-1]
        kernel_size = kernel_dim ** 2

        output_dim = input_dim - kernel_dim + 1
        output_size = output_dim ** 2

        indices = self.create_conv_mat_indices(input_shape[-1], kernel_dim)

        # flatten kernel then repeat it for output_size amount of times
        flat_kernel = tf.reshape(kernel, kernel_size)
        kernel_tile = tf.tile(flat_kernel, [output_size])

        st = tf.SparseTensor(indices=indices, values=kernel_tile, dense_shape=[output_size, input_size])
        st1 = tf.sparse.to_dense(st)
        return st1

    @staticmethod
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
