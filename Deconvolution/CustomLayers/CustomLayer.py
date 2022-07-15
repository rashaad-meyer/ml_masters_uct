import tensorflow as tf
from tensorflow.keras import layers


class CustomLayer(layers.Layer):

    def __init__(self, h_shape, in_channels, out_channels):
        super(CustomLayer, self).__init__()
        self.h_shape = list(h_shape)
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Initialise filter (w) except for the first element
        # So that first element is not trainable
        self.w = tf.random.uniform((1, h_shape[-2] * h_shape[-1] - 1))
        self.w = tf.Variable(self.w, trainable=True)

    def custom_op(self, xm):
        pad_w = tf.constant([[0, 0], [1, 0]])
        # Add 1 to start of filter then reshape to
        # filter shape compatible with conv op
        w0 = tf.pad(self.w, pad_w, mode='CONSTANT', constant_values=1)
        filter_shape = self.h_shape
        filter_shape.append(self.in_channels)
        filter_shape.append(self.out_channels)
        # filter_shape : A 4-D tensor of shape [filter_height, filter_width, in_channels, out_channels]
        # filter_shape needed for conv2d
        w0 = tf.reshape(w0, filter_shape)

        ym = tf.nn.conv2d(xm, w0, strides=[1, 1, 1, 1], padding='same')

        # Get mean value for pixels
        ymean = tf.reduce_mean(ym, -2)
        ymean = tf.reduce_mean(ymean, -2)

        return ymean

    def call(self, inputs):
        return self.custom_op(inputs)


if __name__ == '__main__':
    x = tf.constant([[[1, 1, 1],
                      [1, 1, 1],
                      [1, 1, 1]],
                     [[2, 2, 2],
                      [2, 2, 2],
                      [2, 2, 2]]])
    x = tf.reshape(x, (2, 3, 3, 1))
    print(x)
    x = tf.reduce_mean(x, -2)
    print(x)

    print(x)
