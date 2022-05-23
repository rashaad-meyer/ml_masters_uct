import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt


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


def forward_pass(xm, hrf):
    # pad input
    padding = tf.constant(
        [[0, 0], [int(xm.shape[-2] / 4), int(xm.shape[-2] / 4)], [int(xm.shape[-1] / 4), int(xm.shape[-1] / 4)]])
    xm = tf.pad(xm, padding, "CONSTANT")

    # pad filter to input size
    paddings = tf.constant([[0, xm.shape[-2] - hrf.shape[-2]], [0, xm.shape[-1] - hrf.shape[-1]]])
    hm1 = tf.pad(hrf, paddings, "CONSTANT")

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


def back_prop(xm, hrf, ym, um):
    # FIXME shouldn't have to calculate G again
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


if __name__ == '__main__':
    # random uniform can't be the issue
    x = tf.random.uniform([6, 6], minval=0)
    # hrf = tf.constant([[0.01, 0.02, 0.03], [0.04, 0.05, 0.06], [0.07, 0.08, 0.09]])
    hrf = tf.random.uniform((3, 3), minval=0)
    h = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)

    X = tf.signal.rfft2d(x)
    x1 = tf.signal.irfft2d(X)

    y = forward_pass(x, hrf)
