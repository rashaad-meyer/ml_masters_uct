import tensorflow as tf
from tensorflow.keras import layers


class DeconvDft2dLayer(layers.Layer):

    def __init__(self, h_shape):
        super(DeconvDft2dLayer, self).__init__()
        self.h_shape = h_shape
        # Initialise filter (w) except for the first column
        # So that first column is not trainable
        wf = tf.zeros((h_shape[-2], h_shape[-1] - 1))
        self.w = tf.Variable(wf, trainable=True)
        self.one = tf.pad([[1.0]], [[0, h_shape[-2] - 1], [0, h_shape[-1] - 1]])
        self.one = tf.Variable(self.one, trainable=False)

    def custom_op(self, xm):
        # makes first element of every row = 1 and not trainable
        pad_w = tf.constant([[0, 0], [1, 0]])
        w0 = tf.pad(self.w, pad_w, mode='CONSTANT')
        w0 = tf.add(w0, self.one)

        padding = tf.constant(
            [[0, 0], [int(xm.shape[-2] / 4), int(xm.shape[-2] / 4)], [int(xm.shape[-1] / 4), int(xm.shape[-1] / 4)]])
        xm = tf.pad(xm, padding, "CONSTANT")

        paddings = tf.constant([[0, xm.shape[-2] - w0.shape[-2]], [0, xm.shape[-1] - w0.shape[-1]]])
        hm1 = tf.pad(w0, paddings, "CONSTANT")

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
