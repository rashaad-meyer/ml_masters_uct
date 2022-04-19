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
        return self.custom_op(inputs)


