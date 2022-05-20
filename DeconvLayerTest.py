import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from DeconvDft2dLayer import DeconvDft2dLayer
import DeconvFFT


def deconv_dft_2d_test():
    # xm = tf.random.uniform((1, 16, 16), minval=0)
    data_len = 20000
    xm = tf.ones((data_len, 32, 32))

    hrf = tf.random.uniform((3, 3), minval=0)
    hrf = tf.constant([[0.01, 0.02, 0.03], [0.04, 0.05, 0.06], [0.07, 0.08, 0.09]])
    ym = DeconvFFT.forward_pass(xm, hrf)

    x_train = xm[:int(data_len / 2)]
    y_train = ym[:int(data_len / 2)]

    x_test = xm[int(data_len / 2):]
    y_test = ym[int(data_len / 2):]

    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(32, 32)),
            DeconvDft2dLayer((3, 3))
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.MeanSquaredError(reduction='sum_over_batch_size'),
        metrics=['MeanSquaredError']
    )
    w0 = model.layers[0].w
    w0_diff = tf.reduce_sum(tf.math.square(hrf-w0))

    print("\n*******************************************")
    print("Total difference before")
    print(w0_diff)
    print("Weights before")
    print(w0)
    print("*******************************************\n")

    model.fit(x_train, y_train, batch_size=10, epochs=10, verbose=2)
    model.evaluate(x_test, y_test, batch_size=10, verbose=2)

    wf = model.layers[0].w
    wf_diff = tf.reduce_sum(tf.math.square(hrf - wf))

    print("\n*******************************************")
    print("Total difference after")
    print(wf_diff)
    print("Weights after")
    print(wf)
    print("*******************************************\n")


def deconv_2d_dft_mnist_test():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(28, 28)),
            # DeconvDft2dLayer((3, 3)),
            layers.Flatten(),
            layers.Dense(10)
        ]
    )

    # model.layers[2].trainable = False
    model.summary()

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        metrics=["accuracy"],
    )

    w0 = model.layers[1].kernel

    model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=2)
    model.evaluate(x_test, y_test, batch_size=32, verbose=2)

    wf = model.layers[1].kernel
    tf.print("Weights before")
    tf.print(w0)
    tf.print("Weights after")
    tf.print(wf)


if __name__ == '__main__':
    deconv_dft_2d_test()
    # test()
