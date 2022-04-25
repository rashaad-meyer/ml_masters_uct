import tensorflow as tf
import DeconvPrototype


def generate_deconv_dataset(h, input_shape):
    x = tf.random.uniform(shape=input_shape, minval=0)
    y = []

    for i in x:
        y.append(DeconvPrototype.forward_fcn(i, h))

    return tf.concat([y], 0), x
    # DeconvPrototype.forward_fcn()


if __name__ == '__main__':
    h = tf.random.uniform(shape=[1], minval=0)
    input_shape = [2, 5]
    tf.print(h)
    y = generate_deconv_dataset(h, input_shape)
    tf.print(y[1])
    tf.print(y[0])
