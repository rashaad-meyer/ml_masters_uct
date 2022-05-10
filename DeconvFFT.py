import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt


def np_tf_FFT_test():
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
    # pad filter to input size
    paddings = tf.constant([[0, xm.shape[-2] - hrf.shape[-2]], [0, xm.shape[-1] - hrf.shape[-1]]])
    hm1 = tf.pad(hrf, paddings, "CONSTANT")

    #
    gm1f = 1 / np.fft.fft2(hm1)
    gm2f = tf.roll(tf.reverse(gm1f, [0]), shift=1, axis=0)
    gm3f = tf.roll(tf.reverse(gm1f, [1]), shift=1, axis=1)
    gm4f = tf.roll(tf.reverse(gm3f, [0]), shift=1, axis=0)
    gmf = gm1f * gm2f * gm3f * gm4f

    ymf = gmf * np.fft.fft2(xm)
    ym = np.fft.ifft2(ymf)

    return tf.cast(ym, dtype=tf.float32)


def back_prop(xm, hrf, ym, um):
    return


if __name__ == '__main__':
    x = tf.ones([6, 6])
    h = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)

    hr = tf.reverse(h, [0])
    hr = tf.roll(h, shift=1, axis=0)
    m = np.arange(3)
    m = m * m

    y = forward_pass(x, h)
    print(y)
