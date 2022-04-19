import tensorflow as tf
from tensorflow.keras import layers


class DeconvLayer(layers.Layer):

    def __init__(self, filter_size):
        super(DeconvLayer, self).__init__()
        self.w = self.add_weight(
            name="kernel",
            shape=(filter_size, 1),
            dtype=tf.float32,
            initializer="random_normal",
            trainable=True,
        )

    def call(self, inputs):
        out = []
        for i in inputs:
            out.append(self.forward_fcn(inputs, self.w))
        out = tf.concat([out], 0)
        return out

    @staticmethod
    def forward_fcn(x, h):
        y = tf.TensorArray(tf.float32, size=x.shape[-1], dynamic_size=False, clear_after_read=False)
        v = []
        for i in range(x.shape[-1]):
            element = tf.constant(0, dtype=tf.float32)
            if i >= h.shape[-1]:
                for j in range(h.shape[-1]):
                    temp = tf.multiply(h[j], v[i - j - 1])
                    element = tf.add(element, temp)
                element = tf.add(element, x[i])
            v.append(element)
            y = y.write(i, element)
        y = y.stack()
        return y
