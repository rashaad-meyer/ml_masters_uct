import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def full_deconv(x, h, hr):
    v = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
    j = 0

    for i in x:
        v = v.write(j, deconv(i, h))
        j = j + 1

    v = v.stack()

    y = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
    j = 0

    for i in v:
        y = y.write(j, deconv(i, hr))
        j = j + 1

    y = y.stack()

    return y


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


def back_prop_linear(xm, hrf, ym, um):
    # hrfd: filter dimension
    hr = tf.reverse(hrf, [0])

    grad_inputs = full_deconv(um, hrf, hr)

    grad_vars = []

    for j in range(um.shape[-1]):

        vm = deconv(ym[j], hrf)

        zero = tf.zeros(vm.shape)
        vmq = tf.concat([vm, zero], 0)

        uyqm1 = tf.TensorArray(tf.float32, size=hrf.shape[-1], dynamic_size=False, clear_after_read=False)

        for i in range(hrf.shape[-1]):
            vmq = tf.roll(vmq, shift=1, axis=0)
            temp = tf.tensordot(um[j], vmq[:xm.shape[-1]], 1)
            uyqm1 = uyqm1.write(i, temp)

        uyqm1 = uyqm1.stack()

        vm = deconv(ym[j], hr)

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


if __name__ == '__main__':
    x0 = tf.random.uniform((20, 20), minval=0)
    h0 = tf.constant([[1, 0.1, 0.1, 0.1, 0.1]])
    h0_r = tf.reverse(h0, [0])

    y0 = full_deconv(x0, h0, h0_r)
    u0 = tf.random.uniform(y0.shape, minval=0, maxval=0.1)

    out1, out2 = back_prop_linear(x0[0], h0, y0, u0)

    tf.print(out1.shape)
    tf.print(out2.shape)
