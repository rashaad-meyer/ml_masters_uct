import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class CustomConvolutionLayer(layers.Layer):
    def __init__(self, kernel_size):
        self.w = self.add_weight(
            name="w",
            shape=(kernel_size, kernel_size),
            initializer="random_normal",
            trainable=True
        )

    def call(self, inputs):
        # Convolve input with kernel
        batch_size = len(inputs)  # FIXME len might be slow so change it
        
        return layers.Conv2D()

