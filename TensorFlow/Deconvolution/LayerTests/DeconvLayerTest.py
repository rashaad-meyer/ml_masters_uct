import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from TensorFlow.Deconvolution.CustomLayers.DeconvDft2dLayer import DeconvDft2dLayer
from TensorFlow.Deconvolution.Utility import DeconvFFT


def deconv_dft_2d_test():
    data_len = 2000
    data_shape = (16, 16)
    h_shape = (3, 3)

    xm = tf.random.uniform((data_len, data_shape[-2], data_shape[-1], 1), minval=0)
    pad_w = tf.constant([[0, 0], [1, 0]])

    hrf = tf.random.uniform((1, h_shape[-2] * h_shape[-1] - 1), minval=0, maxval=0.5)

    ym = DeconvFFT.forward_pass(xm, hrf, (3, 3))
    print(tf.reduce_max(ym))

    x_train = xm[:int(data_len / 2)]
    y_train = ym[:int(data_len / 2)]

    x_test = xm[int(data_len / 2):]
    y_test = ym[int(data_len / 2):]

    nn_input_shape = list(data_shape)
    nn_input_shape.append(1)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=nn_input_shape),
            DeconvDft2dLayer(h_shape),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9),
        loss=tf.keras.losses.MeanAbsoluteError()
    )

    model.summary()

    w0 = model.layers[0].w
    w0_diff = tf.reduce_sum(tf.math.square(hrf - w0))

    print("\n-------------------------------------------")
    print("Total difference before")
    print(w0_diff)
    print("Weights before")
    print(w0)
    print("-------------------------------------------\n")

    model.fit(x_train, y_train, batch_size=10, epochs=20, verbose=2)
    model.evaluate(x_test, y_test, batch_size=10, verbose=2)

    wf = model.layers[0].w
    wf_diff = tf.reduce_sum(tf.math.square(hrf - wf))

    print("\n-------------------------------------------")
    print("Total difference after")
    print(wf_diff)
    print("Weights after")
    print(wf)
    print("-------------------------------------------\n")

    print("\n-------------------------------------------")
    print("Target filter")
    print(hrf)
    print("-------------------------------------------\n")
    """
    plt.subplot(211)
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
     plot mse during training
    plt.subplot(212)
    plt.title('Mean Squared Error')
    plt.plot(history.history['mean_squared_error'], label='train')
    plt.plot(history.history['val_mean_squared_error'], label='test')
    plt.legend()
    plt.show()
    """


def deconv_2d_dft_mnist_test():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(28, 28)),
            DeconvDft2dLayer((3, 3)),
            layers.Flatten(),
            layers.Dense(10)
        ]
    )

    model.layers[2].trainable = False
    model.summary()

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
        metrics=["accuracy"],
    )

    w0 = model.layers[0].w
    print("Weights before")
    print(w0)

    model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=2)
    model.evaluate(x_test, y_test, batch_size=32, verbose=2)

    wf = model.layers[0].w

    print("Weights after")
    print(wf)


if __name__ == '__main__':
    deconv_dft_2d_test()
    # deconv_2d_dft_mnist_test()
