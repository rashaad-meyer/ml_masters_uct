import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
from keras.datasets import cifar10


def np_tf_fft_test():
    sr = 2000
    ts = 1.0 / sr
    t = np.arange(0, 1, ts)

    x = 3 * np.sin(2 * np.pi * t)
    x += np.sin(2 * np.pi * 4 * t)
    x += 0.5 * np.sin(2 * np.pi * 7 * t)

    X = np.fft.fft(x)

    N = len(X)
    n = np.arange(N)
    T = N / sr
    freq = n / T

    X_abs = np.abs(X)

    plt.figure(figsize=(12, 6))
    plt.subplot(121)

    plt.stem(freq, np.abs(X))
    plt.xlabel('Freq (Hz)')
    plt.ylabel('NP FFT Amplitude |X(freq)|')
    plt.xlim(0, 10)

    t = tf.range(0, 1, ts)
    t = tf.cast(t, tf.complex128)

    x1 = 3 * tf.math.sin(2 * math.pi * 1 * t)
    x2 = 1 * tf.math.sin(2 * math.pi * 4 * t)
    x3 = 2 * tf.math.sin(2 * math.pi * 7 * t)

    x = tf.add(tf.add(x1, x2), x3)

    X = tf.signal.fft(x)

    N = x.shape[-1]
    n = tf.range(N, dtype=tf.float32)
    T = N / sr
    freq = n / T

    plt.subplot(122)
    plt.stem(freq, tf.math.abs(X))
    plt.xlabel('Freq (Hz)')
    plt.ylabel('TF FFT Amplitude |X(freq)|')
    plt.xlim(0, 10)
    plt.show()


def forward_pass(xm, hrf, h_shape):
    # pad input

    xm = tf.reshape(xm, (-1, xm.shape[-3], xm.shape[-2]))

    pad_w = tf.constant([[0, 0], [1, 0]])
    # Set first element to 1 then reshape into specified filter shape
    w0 = tf.pad(hrf, pad_w, mode='CONSTANT', constant_values=1)
    w0 = tf.reshape(w0, h_shape)

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

    ym = tf.reshape(ym, (-1, ym.shape[-2], ym.shape[-1], 1))
    ym = tf.image.central_crop(ym, 0.67)

    return ym


def back_prop(xm, hrf, ym, um):
    paddings = tf.constant([[0, xm.shape[-2] - hrf.shape[-2]], [0, xm.shape[-1] - hrf.shape[-1]]])
    hm1 = tf.pad(hrf, paddings, "CONSTANT")

    # Calculate G
    gm1f = 1 / np.fft.fft2(hm1)
    gm2f = tf.roll(tf.reverse(gm1f, [0]), shift=1, axis=0)
    gm3f = tf.roll(tf.reverse(gm1f, [1]), shift=1, axis=1)
    gm4f = tf.roll(tf.reverse(gm3f, [0]), shift=1, axis=0)
    gmf = gm1f * gm2f * gm3f * gm4f

    umf = np.fft.fft2(um)
    uypxm = np.fft.ifft2(gmf * umf)

    return uypxm


def initialise_w(h_shape):
    wf = tf.TensorArray(tf.float32, size=h_shape[-2], dynamic_size=False)
    for i in range(h_shape[-2]):
        wa = tf.TensorArray(tf.float32, size=h_shape[-1], dynamic_size=False)
        for j in range(h_shape[-1]):
            v = tf.Variable([0.0], trainable=True)

            if i == 0 and j == 0:
                v = tf.Variable([1.0], trainable=False)

            wa = wa.write(j, v)

        wa = wa.stack()
        wf = wf.write(i, wa)
    wf = wf.stack()
    w = tf.reshape(wf, h_shape)
    return w


def forward_pass_multichannel(xm, w, h_shape):
    xm = tf.transpose(xm, perm=[0, 3, 1, 2])

    # Transform filter into right shape
    pad_w = tf.constant([[0, 0], [1, 0]])
    w0 = tf.pad(w, pad_w, mode='CONSTANT', constant_values=1)
    w0 = tf.reshape(w0, h_shape)
    print(w0)

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


if __name__ == '__main__':
    # Define filter size and initialise trainable parameters
    h_shape = (3, 3, 3)
    w = tf.random.uniform((h_shape[-1], h_shape[-3] * h_shape[-2] - 1))
    print(w)

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.moveaxis(x_train, -1, 1)
    x_test = np.moveaxis(x_test, -1, 1)
    ym = forward_pass_multichannel(x_test[:2], w, h_shape)

