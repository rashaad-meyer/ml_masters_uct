import tensorflow as tf
import DeconvPrototype


def generate_deconv_dataset(h, input_shape):
    x = tf.random.uniform(shape=input_shape, minval=0)
    y = forward(x, h)
    return x, y


def forward(inputs, h):
    v = []
    for i in inputs:
        v.append(DeconvPrototype.forward_fcn(i, h))
    v = tf.concat([v], 0)

    y = []
    for i in v:
        y.append(DeconvPrototype.forward_fcn(i, h))
    y = tf.concat([y], 0)

    return y


if __name__ == '__main__':
    h = tf.random.uniform(shape=[1, 8], minval=0)
    input_shape = [10, 32]
    tf.print(h)
    (x_train, y_train) = generate_deconv_dataset(h, input_shape)
    tf.print(x_train)
    tf.print(y_train)
