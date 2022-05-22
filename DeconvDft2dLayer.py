import tensorflow as tf
from tensorflow.keras import layers


class DeconvDft2dLayer(layers.Layer):

    def __init__(self, h_shape):
        super(DeconvDft2dLayer, self).__init__()
        self.h_shape = h_shape
        self.w = tf.Variable(name='w',
                             initial_value=tf.random.uniform(h_shape, minval=0, maxval=0.1),
                             trainable=True)

    def custom_op(self, xm):
        padding = tf.constant(
            [[0, 0], [int(xm.shape[-2] / 4), int(xm.shape[-2] / 4)], [int(xm.shape[-1] / 4), int(xm.shape[-1] / 4)]])
        xm = tf.pad(xm, padding, "CONSTANT")

        paddings = tf.constant([[0, xm.shape[-2] - self.w.shape[-2]], [0, xm.shape[-1] - self.w.shape[-1]]])
        hm1 = tf.pad(self.w, paddings, "CONSTANT")

        gm1f = tf.divide(1, tf.signal.rfft2d(hm1))
        gm2f = tf.roll(tf.reverse(gm1f, [0]), shift=1, axis=0)
        gm3f = tf.roll(tf.reverse(gm1f, [1]), shift=1, axis=1)
        gm4f = tf.roll(tf.reverse(gm3f, [0]), shift=1, axis=0)

        gmf1 = tf.multiply(gm1f, gm2f)
        gmf2 = tf.multiply(gm3f, gm4f)
        gmf = tf.multiply(gmf1, gmf2)

        ymf = tf.multiply(gmf, tf.signal.rfft2d(xm))
        ym = tf.signal.irfft2d(ymf)

        return ym

    def call(self, inputs):
        return self.custom_op(inputs)
