import tensorflow as tf
from tensorflow.keras import layers


class DeconvDft2dRgbLayer(layers.Layer):

    def __init__(self, h_shape):
        super(DeconvDft2dRgbLayer, self).__init__()
        self.h_shape = h_shape
        # TODO make more robust by inferring num_channels from input (can use __build__)
        # Initialise filter (w) except for the first element
        # So that first element is not trainable
        self.w = tf.random.uniform((h_shape[-1], h_shape[-3] * h_shape[-2] - 1, 1), maxval=0.1)
        self.w = tf.Variable(self.w, trainable=True)

    def custom_op(self, xm):
        # put RGB dimensions after batch_size
        xm = tf.transpose(xm, perm=[0, 3, 1, 2])

        pad_w = tf.constant([[0, 0], [1, 0], [0, 0]])
        w0 = tf.pad(self.w, pad_w, mode='CONSTANT', constant_values=1)
        w0 = tf.reshape(w0, self.h_shape)

        padding = tf.constant(
            [[0, 0], [0, 0], [int(xm.shape[-2] / 4), int(xm.shape[-2] / 4)],
             [int(xm.shape[-1] / 4), int(xm.shape[-1] / 4)]])
        xm = tf.pad(xm, padding, "CONSTANT")

        paddings = tf.constant([[0, 0], [0, xm.shape[-2] - w0.shape[-2]], [0, xm.shape[-1] - w0.shape[-1]]])
        hm1 = tf.pad(w0, paddings, "CONSTANT")

        gm1f = tf.divide(1, tf.signal.rfft2d(hm1))
        gm2f = tf.roll(tf.reverse(gm1f, [-2]), shift=1, axis=-2)
        gm3f = tf.roll(tf.reverse(gm1f, [-1]), shift=1, axis=-1)
        gm4f = tf.roll(tf.reverse(gm3f, [-2]), shift=1, axis=-2)

        gmf1 = tf.multiply(gm1f, gm2f)
        gmf2 = tf.multiply(gm3f, gm4f)
        gmf = tf.multiply(gmf1, gmf2)

        ymf = tf.multiply(gmf, tf.signal.rfft2d(xm))
        ym = tf.signal.irfft2d(ymf)

        ym = tf.transpose(ym, perm=[0, 2, 3, 1])
        return ym

    def call(self, inputs):
        return self.custom_op(inputs)
