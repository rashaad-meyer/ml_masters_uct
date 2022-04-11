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

        # create toeplitz-like matrix
        input_dim = tf.cast(input_shape[-1], tf.int32)
        input_size = input_dim * input_dim

        output_dim = input_dim - self.kernel_size + 1
        output_size = output_dim * output_dim

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

        self.w = tf.Variable(tf.concat([temp], 0), trainable=True)

    @tf.custom_gradient
    def custom_op(self, inputs):
        result = tf.transpose(tf.matmul(self.w, inputs, transpose_b=True))

        def custom_grad(dy, variable):
            return dy

        return result, custom_grad

    def call(self, inputs):
        return tf.transpose(tf.matmul(self.w, inputs, transpose_b=True))
