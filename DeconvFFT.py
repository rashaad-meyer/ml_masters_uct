import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt

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


def forward_pass(x):
    return
