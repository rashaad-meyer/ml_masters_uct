import tensorflow as tf
from tensorflow.keras import layers


class DeconvLayer(layers.Layer):

    def __init__(self, filter_size):
        super(DeconvLayer, self).__init__()
        self.kernel = self.add_weight(
            name="kernel",
            shape=(1, filter_size),
            dtype=tf.float32,
            initializer="random_normal",
            trainable=True,
        )
        self.kernel_r = tf.reverse(self.kernel, [-1])

    def call(self, inputs):
        return self.custom_op(inputs, self.kernel, self.kernel_r)

    @tf.custom_gradient
    def custom_op(self, inputs, h, hr):
        y = self.full_deconv(inputs, h, hr)

        h = self.kernel

        def grad_fn(dy, variables):

            assert variables is not None
            assert len(variables) == 1

            grad_vars = []

            grad_inputs = self.full_deconv(inputs, h, hr)

            return grad_inputs

        return y, grad_fn

    def full_deconv(self, inputs, h, hr):
        v = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        j = 0

        for i in inputs:
            v = v.write(j, self.deconv(i, h))
            j = j + 1

        v = v.stack()

        y = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        j = 0

        for i in v:
            y = y.write(j, self.deconv(i, hr))
            j = j + 1

        y = y.stack()

        return y

    @staticmethod
    def deconv(x, h):
        y = tf.TensorArray(tf.float32, size=x.shape[-1], dynamic_size=False, clear_after_read=False)
        v = []
        for i in range(x.shape[-1]):
            element = tf.constant(0, dtype=tf.float32)
            if i >= h.shape[-1]:
                for j in range(h.shape[-1]):
                    temp = tf.multiply(h[0][j], v[i - j - 1])
                    element = tf.add(element, temp)
                element = tf.add(element, x[i])
            v.append(element)
            y = y.write(i, element)
        y = y.stack()
        return y
