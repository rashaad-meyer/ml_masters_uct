import tensorflow as tf
from Deconvolution import DeconvPrototype


def generate_deconv_dataset(h, input_shape):
    x = tf.random.uniform(shape=input_shape, minval=0)
    hr = tf.reverse(h, [1])

    y = DeconvPrototype.full_deconv(x, h, hr)
    return x, y


if __name__ == '__main__':
    h = tf.random.uniform(shape=[1, 8], minval=0)
    tf.print(h, summarize=8)
    input_shape = [10, 32]
    tf.print(h)
    (x_train, y_train) = generate_deconv_dataset(h, input_shape)
    tf.print(x_train.shape)
    tf.print(y_train.shape)
