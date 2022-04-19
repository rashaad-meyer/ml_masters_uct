import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def forward_fcn(x, h):
    # TODO change to use tensors
    y = tf.TensorArray(tf.float32, size=x.shape[-1], dynamic_size=False, clear_after_read=False)
    for i in range(len(x)):
        y.append(0)
        if i >= len(h):
            for j in range(len(h)):
                y[i] = y[i] + h[j]*y[i-j-1]
            y[i] = y[i] + x[i]
    return y


if __name__ == '__main__':
    x_ = [1, 2, 3, 4, 5, 6]
    h_ = [1, 2, 3]
    y_ = forward_fcn(x_, h_)
    print(x_)
    print(h_)
    print(y_)
