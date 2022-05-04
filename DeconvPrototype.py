import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def forward_fcn(x, h):
    y = tf.TensorArray(tf.float32, size=x.shape[-1], dynamic_size=False, clear_after_read=False)
    v = []
    for i in range(x.shape[-1]):
        element = tf.constant(0, dtype=tf.float32)
        if i >= h.shape[-1]:
            for j in range(h.shape[-1]):
                temp = tf.multiply(h[0][j], v[i-j-1])
                element = tf.add(element, temp)
            element = tf.add(element, x[i])
        v.append(element)
        y = y.write(i, element)
    y = y.stack()
    return y


if __name__ == '__main__':
    # x_ = [1, 2, 3, 4, 5, 6]
    # h_ = [1, 2, 3]
    x_ = tf.constant([1, 2, 3, 4, 5, 6], dtype=tf.float32)
    h_ = tf.constant([[1, 2, 3]], dtype=tf.float32)

    print(h_[0])

    y_ = forward_fcn(x_, h_)
    print(x_)
    print(h_)
    print(y_)

    # for i in range(x_.shape[-1] - h_.shape[-1]):
    #     print(x_[1:3, :-1])

