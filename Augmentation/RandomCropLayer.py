import random

import tensorflow as tf

from tensorflow.keras import layers


class RandomCrop(layers.Layer):
    def __init__(self, crop_size):
        super().__init__()
        self.crop_norm = None
        self.crop_size = crop_size

    def build(self, input_shape):
        self.crop_norm = (self.crop_size[-2] / input_shape[-3], self.crop_size[-1] / input_shape[-2])

    def call(self, inputs):
        diff_x = (1 - self.crop_norm[-1]) / 2
        diff_y = (1 - self.crop_norm[-2]) / 2

        offset_x1 = tf.add(tf.random.uniform((1, inputs.shape[0]), minval=-0.05, maxval=0.05), diff_x)
        offset_y1 = tf.add(tf.random.uniform((1, inputs.shape[0]), minval=-0.05, maxval=0.05), diff_y)
        offset_x2 = tf.add(offset_x1, self.crop_norm[-1])
        offset_y2 = tf.add(offset_y1, self.crop_norm[-2])

        boxes = tf.transpose(tf.concat([offset_y1, offset_x1, offset_y2, offset_x2], 0))

        indices = tf.range(inputs.shape[0])

        inputs = tf.image.crop_and_resize(inputs, boxes, indices, self.crop_size)
        return inputs
