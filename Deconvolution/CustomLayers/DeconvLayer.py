import tensorflow as tf
from tensorflow.keras import layers


class DeconvLayer(layers.Layer):

    def __init__(self, filter_size):
        super(DeconvLayer, self).__init__()
        self.kernel = self.add_weight(
            name="kernel",
            shape=(1, filter_size),
            dtype=tf.float32,
            initializer="random_normal",
            trainable=True,
        )
        self.kernel_r = tf.reverse(self.kernel, [-1])

    def call(self, inputs):
        return self.custom_op(inputs, self.kernel, self.kernel_r)

    @tf.custom_gradient
    def custom_op(self, xm, hrf, hr):
        ym = self.full_deconv(xm, hrf, hr)

        def grad_fn(um, variables):
            assert variables is not None
            assert len(variables) == 1

            grad_inputs = self.full_deconv(um, hrf, hr)

            grad_vars = []

            for j in range(um.shape[-1]):

                vm = self.deconv(ym[j], hrf)

                zero = tf.zeros(vm.shape)
                vmq = tf.concat([vm, zero], 0)

                uyqm1 = tf.TensorArray(tf.float32, size=hrf.shape[-1], dynamic_size=False, clear_after_read=False)

                for i in range(hrf.shape[-1]):
                    vmq = tf.roll(vmq, shift=1, axis=0)
                    temp = tf.tensordot(um[j], vmq[:xm.shape[-1]], 1)
                    uyqm1 = uyqm1.write(i, temp)

                uyqm1 = uyqm1.stack()

                vm = self.deconv(ym[j], hr)

                zero = tf.zeros(vm.shape)
                vmq = tf.concat([vm, zero], 0)

                uyqm2 = tf.TensorArray(tf.float32, size=hrf.shape[-1], dynamic_size=False, clear_after_read=False)

                for i in range(hrf.shape[-1]):
                    vmq = tf.roll(vmq, shift=-1, axis=0)
                    temp = tf.tensordot(um[j], vmq[:xm.shape[-1]], 1)
                    uyqm2 = uyqm2.write(i, temp)

                uyqm2 = uyqm2.stack()

                uyqm = -tf.add(uyqm1, uyqm2)
                grad_vars.append(uyqm)

            grad_vars = tf.concat([grad_vars], 0)
            return grad_inputs, grad_vars

        return ym, grad_fn

    def full_deconv(self, inputs, h, hr):
        v = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        j = 0

        for i in inputs:
            v = v.write(j, self.deconv(i, h))
            j = j + 1

        v = v.stack()

        y = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        j = 0

        for i in v:
            y = y.write(j, self.deconv(i, hr))
            j = j + 1

        y = y.stack()

        return y

    @staticmethod
    def deconv(x, h):
        y = tf.TensorArray(tf.float32, size=x.shape[-1], dynamic_size=False, clear_after_read=False)
        for i in range(x.shape[-1]):
            element = tf.constant(0, dtype=tf.float32)
            if i >= h.shape[-1]:
                for j in range(h.shape[-1]):
                    temp = tf.multiply(h[0][j], x[i - j - 1])
                    element = tf.add(element, temp)
                element = tf.add(element, x[i])
            y = y.write(i, element)
        y = y.stack()
        return y
