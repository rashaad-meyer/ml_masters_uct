import tensorflow as tf
from keras import models
from keras import layers
import numpy as np


@tf.custom_gradient
def custom_operation(w, b, x):
    y = tf.matmul(x, w) + b

    def grad_fn(dldz):
        wt = tf.transpose(w)
        xt = tf.transpose(x)

        dldw = tf.matmul(xt, dldz)
        dldx = tf.matmul(dldz, wt)

        dldb = tf.reduce_sum(dldz, axis=0)

        return dldw, dldb, dldx

    return y, grad_fn


# end def custom_op

# Generate training data
W = tf.random.uniform((2, 3))
x_train = np.array(tf.random.uniform((10, 2)))
y_train = np.array(tf.matmul(x_train, W))
y_train = y_train + 0.01 * tf.random.normal(tf.shape(y_train))


class CustomLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(CustomLayer, self).__init__()
        self.w = None
        self.b = None

    def build(self, input_shape):
        self.w = self.add_weight(
            name='w',
            shape=(2, 3),
            initializer='random_normal',
            trainable=True
        )
        self.b = self.add_weight(
            name='b',
            shape=(3,),
            initializer='random_normal',
            trainable=True
        )

    def call(self, x):
        return custom_operation(self.w, self.b, x)


inputs = layers.Input(2)
outputs = CustomLayer()(inputs)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=tf.keras.optimizers.SGD(),
              loss=tf.keras.losses.MeanSquaredError())
model.fit(x_train, y_train, batch_size=5, epochs=5, verbose=2)

