import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np


@tf.custom_gradient
def custom_op(xm, w, h_shape):
    xm = tf.reshape(xm, (-1, xm.shape[-3], xm.shape[-2]))
    paddings = tf.constant([[0, xm.shape[-2] - w.shape[-2]],
                            [0, xm.shape[-1] - w.shape[-1]]])
    hm1 = tf.pad(w, paddings, "CONSTANT")

    xm = tf.cast(xm, dtype=tf.complex64)
    hm1 = tf.cast(hm1, dtype=tf.complex64)

    gm1f = tf.divide(1, tf.signal.fft2d(hm1))
    gm2f = tf.roll(tf.reverse(gm1f, [0]), shift=1, axis=0)
    gm3f = tf.roll(tf.reverse(gm1f, [1]), shift=1, axis=1)
    gm4f = tf.roll(tf.reverse(gm3f, [0]), shift=1, axis=0)

    gmf1 = tf.multiply(gm1f, gm2f)
    gmf2 = tf.multiply(gm3f, gm4f)
    gmf = tf.multiply(gmf1, gmf2)

    ymf = tf.multiply(gmf, tf.signal.fft2d(xm))
    ym = tf.signal.ifft2d(ymf)
    ym = tf.cast(ym, dtype=tf.float32)

    ym = tf.reshape(ym, (-1, ym.shape[-2], ym.shape[-1], 1))

    def grad_fn(um):
        um = tf.reshape(um, (-1, um.shape[-3], um.shape[-2]))
        umi = tf.cast(um, dtype=tf.complex64)

        # backprop layer inputs

        umf = tf.signal.fft2d(umi)
        dldxf = tf.multiply(gmf, umf)
        dldx = tf.signal.ifft2d(dldxf)
        dldx = tf.cast(dldx, dtype=tf.float32)

        hsir_top = tf.zeros((1, h_shape[0] * h_shape[1]), dtype=tf.int32)
        hsir_bot = tf.range(0, h_shape[0] * h_shape[1], dtype=tf.int32)
        hsir_bot = tf.reshape(hsir_bot, (1, -1))
        hsir = tf.concat([hsir_top, hsir_bot], 0)
        hsir = hsir[:, 1:]

        dldw = tf.zeros((1, hsir.shape[1]))

        vm = tf.signal.ifft2d(tf.multiply(gm1f, ymf))
        vm = tf.cast(vm, tf.float32)

        # g1
        hsirf = hsir
        for j in range(hsirf.shape[1]):
            j_zeros = np.zeros((1, hsirf.shape[1]), dtype=np.float32)
            j_zeros[0][j] = 1
            vmq = tf.roll(vm, shift=hsirf[1][j], axis=0)
            temp = tf.multiply(vmq, um)
            temp = tf.multiply(j_zeros, tf.reduce_sum(temp))
            dldw = tf.add(dldw, temp)

        # g2
        hsirf = hsir[:, :]
        hsirf = -hsirf

        for j in range(hsirf.shape[1]):
            j_zeros = np.zeros((1, hsirf.shape[1]), dtype=np.float32)
            j_zeros[0][j] = 1
            vmq = tf.roll(vm, shift=hsirf[1][j], axis=0)
            temp = tf.multiply(vmq, um)
            temp = tf.multiply(j_zeros, tf.reduce_sum(temp))
            dldw = tf.add(dldw, temp)

        # g3
        hsirf = hsir
        hsirf = -hsirf
        for j in range(hsirf.shape[1]):
            j_zeros = np.zeros((1, hsirf.shape[1]), dtype=np.float32)
            j_zeros[0][j] = 1
            vmq = tf.roll(vm, shift=hsirf[1][j], axis=0)
            temp = tf.multiply(vmq, um)
            temp = tf.multiply(j_zeros, tf.reduce_sum(temp))
            dldw = tf.add(dldw, temp)

        # g4
        hsirf = hsir
        hsirf = -hsirf
        for j in range(hsirf.shape[1]):
            j_zeros = np.zeros((1, hsirf.shape[1]), dtype=np.float32)
            j_zeros[0][j] = 1
            vmq = tf.roll(vm, shift=hsirf[1][j], axis=0)
            temp = tf.multiply(vmq, um)
            temp = tf.multiply(j_zeros, tf.reduce_sum(temp))
            dldw = tf.add(dldw, temp)

        return dldx, dldw, None, None

    return ym, grad_fn


class DeconvDft2dLayer(layers.Layer):

    def __init__(self, h_shape, pad_amount=0.5):
        super(DeconvDft2dLayer, self).__init__()
        self.w = None
        self.h_shape = h_shape
        self.pad_amount = pad_amount

    def build(self, input_shape):
        # Initialise filter (w) except for the first element
        # So that first element is not trainable
        # Randomly initialise other components and multiply by factor of 1/2*sqrt(no. of pixels)
        self.w = tf.random.uniform((1, self.h_shape[-2] * self.h_shape[-1] - 1))
        self.w = tf.Variable(self.w, trainable=True)

    def call(self, inputs):
        return custom_op(inputs, self.w, self.h_shape)


if __name__ == '__main__':
    print('-- CUSTOM LAYER TEST --')
    from keras.datasets import mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    model = keras.Sequential([
        keras.Input(shape=(28, 28, 1)),
        DeconvDft2dLayer((2, 2)),
        layers.Flatten(),
        layers.Dense(10)
    ])

    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=2)
    model.evaluate(x_test, y_test, batch_size=32, verbose=2)
