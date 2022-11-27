import numpy as np
import tensorflow as tf
from Deconvolution.CustomLayers.DeconvDft2dCustomGradLayer import custom_op


def check_gradient():
    # initialise variables needed for forward pass
    h_shape = (2, 2)
    w = tf.random.uniform((1, h_shape[-2] * h_shape[-1] - 1), seed=42)
    w = tf.Variable(w)
    xm = tf.random.uniform((1, 16, 16, 1), seed=42)
    pad_amount = 0.5

    # calculate gradient approximation
    grad_approx = get_grad_approx(h_shape, pad_amount, w, xm)

    print('Gradient using gradient checking:')
    print(grad_approx)

    # Calculate forward pass and get gradient tape to keep track of operations
    with tf.GradientTape() as tape:
        loss = forward_prop(h_shape, pad_amount, w, xm)

    # get gradients from GradientTape
    grad = tape.gradient(loss, w)

    print('\nGradient using gradient autodiff:')
    print(grad)

    grad = np.array(grad, dtype=np.float32)
    grad_approx = np.array(grad_approx, dtype=np.float32)

    difference = np.linalg.norm(grad - grad_approx) / (np.linalg.norm(grad) + np.linalg.norm(grad_approx))

    if difference > 1e-7:
        print(
            "\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
    else:
        print(
            "\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")


@tf.function
def forward_prop(h_shape, pad_amount, w, xm):
    with tf.GradientTape() as tape:
        xm = tf.reshape(xm, (-1, xm.shape[-3], xm.shape[-2]))

        pad_w = tf.constant([[0, 0], [1, 0]])
        # Set first element to 1 then reshape into specified filter shape
        w0 = tf.pad(w, pad_w, mode='CONSTANT', constant_values=1)
        w0 = tf.reshape(w0, h_shape)

        padding = tf.constant(
            [[0, 0], [int(xm.shape[-2] * pad_amount), int(xm.shape[-2] * pad_amount)],
             [int(xm.shape[-1] * pad_amount), int(xm.shape[-1] * pad_amount)]])

        xm = tf.pad(xm, padding, "CONSTANT")
        paddings = tf.constant([[0, xm.shape[-2] - w0.shape[-2]], [0, xm.shape[-1] - w0.shape[-1]]])

        hm1 = tf.pad(w0, paddings, "CONSTANT")

        gm1f = tf.divide(1, tf.signal.rfft2d(hm1))
        gm2f = tf.roll(tf.reverse(gm1f, [0]), shift=1, axis=0)
        gm3f = tf.roll(tf.reverse(gm1f, [1]), shift=1, axis=1)
        gm4f = tf.roll(tf.reverse(gm3f, [0]), shift=1, axis=0)

        gmf1 = tf.multiply(gm1f, gm2f)
        gmf2 = tf.multiply(gm3f, gm4f)
        gmf = tf.multiply(gmf1, gmf2)
        ymf = tf.multiply(gmf, tf.signal.rfft2d(xm))

        ym = tf.signal.irfft2d(ymf)
        ym = tf.reshape(ym, (-1, ym.shape[-2], ym.shape[-1], 1))

        # crop to original image size
        ym = tf.image.central_crop(ym, 1 / (1 + 2 * pad_amount))

        loss = tf.reduce_mean(ym ** 2)
    return loss


def get_grad_approx(h_shape, pad_amount, w, xm, eps=1e-7):
    grad_approx = np.zeros(w.shape[1])

    for i in range(w.shape[1]):
        # Reset w_plus and w_minus
        w_plus = np.array(w)
        w_minus = np.array(w)

        # add and minus epsilon from 1 element in kernel
        w_plus[0][i] = w[0][i] + eps
        w_minus[0][i] = w[0][i] - eps

        # calculate loss with new w
        j_plus = forward_prop(h_shape, pad_amount, w_plus, xm)
        j_minus = forward_prop(h_shape, pad_amount, w_minus, xm)

        # find gradient
        grad_approx[i] = (j_plus - j_minus) / (2 * eps)

    # reshape to grad_approx to kernel shape
    grad_approx = tf.reshape(grad_approx, w.shape)

    return grad_approx


def custom_grad_check():
    # Expected output:
    # 102.2888  113.4538  115.5920  125.2290  118.3843   74.5657   98.5300
    h_shape = (2, 4)

    X = tf.range(0, h_shape[0])
    Y = tf.range(0, h_shape[1])
    X1, X2 = tf.meshgrid(X, Y)
    X1 = tf.reshape(X1, (1, -1))
    X2 = tf.reshape(X2, (1, -1))

    hsir = {}

    hsirf = tf.concat([X1, X2], axis=0)

    hsir['G1'] = hsirf[:, 1:]
    hsir['G2'] = tf.concat([-X1, X2], axis=0)[:, 1:]
    hsir['G3'] = tf.concat([X1, -X2], axis=0)[:, 1:]
    hsir['G4'] = tf.concat([-X1, -X2], axis=0)[:, 1:]

    tf.random.set_seed(42)

    w = tf.constant([[1, 0.1, 0.2, -0.1],
                  [0, 0.1, 0.3, 0]])


    xm = tf.random.uniform((1, 28, 28), seed=42)
    um = tf.random.uniform((1, 28, 28))

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

    def grad_fn(um):
        # um = tf.reshape(um, (-1, um.shape[-3], um.shape[-2]))
        umi = tf.cast(um, dtype=tf.complex64)

        # backprop layer inputs

        umf = tf.signal.fft2d(umi)
        dldxf = tf.multiply(gmf, umf)
        dldx = tf.signal.ifft2d(dldxf)
        dldx = tf.cast(dldx, dtype=tf.float32)

        dldw = tf.zeros((1, hsir['G1'].shape[1]))

        vm = tf.signal.ifft2d(tf.multiply(gm1f, ymf))
        vm = tf.cast(vm, tf.float32)

        # G1
        hsirf = hsir['G1']

        vm = tf.signal.ifft2d(tf.multiply(gm1f, ymf))
        vm = tf.cast(vm, tf.float32)

        for j in range(hsirf.shape[1]):
            j_zeros = np.zeros((1, hsirf.shape[1]), dtype=np.float32)
            j_zeros[0][j] = 1
            shift = tf.transpose(hsirf[:, j])
            vmq = tf.roll(vm, shift=shift, axis=[1, 2])
            temp = tf.multiply(vmq, um)
            temp = tf.multiply(j_zeros, tf.reduce_sum(temp))
            dldw = tf.add(dldw, temp)

        # G2
        hsirf = hsir['G2']

        vm = tf.signal.ifft2d(tf.multiply(gm2f, ymf))
        vm = tf.cast(vm, tf.float32)

        for j in range(hsirf.shape[1]):
            j_zeros = np.zeros((1, hsirf.shape[1]), dtype=np.float32)
            j_zeros[0][j] = 1
            shift = tf.transpose(hsirf[:, j])
            vmq = tf.roll(vm, shift=shift, axis=[1, 2])
            temp = tf.multiply(vmq, um)
            temp = tf.multiply(j_zeros, tf.reduce_sum(temp))
            dldw = tf.add(dldw, temp)

        # G3
        hsirf = hsir['G3']

        vm = tf.signal.ifft2d(tf.multiply(gm3f, ymf))
        vm = tf.cast(vm, tf.float32)

        for j in range(hsirf.shape[1]):
            j_zeros = np.zeros((1, hsirf.shape[1]), dtype=np.float32)
            j_zeros[0][j] = 1
            shift = tf.transpose(hsirf[:, j])
            vmq = tf.roll(vm, shift=shift, axis=[1, 2])
            temp = tf.multiply(vmq, um)
            temp = tf.multiply(j_zeros, tf.reduce_sum(temp))
            dldw = tf.add(dldw, temp)

        # G4
        hsirf = hsir['G4']

        vm = tf.signal.ifft2d(tf.multiply(gm4f, ymf))
        vm = tf.cast(vm, tf.float32)

        for j in range(hsirf.shape[1]):
            j_zeros = np.zeros((1, hsirf.shape[1]), dtype=np.float32)
            j_zeros[0][j] = 1
            shift = tf.transpose(hsirf[:, j])
            vmq = tf.roll(vm, shift=shift, axis=[1, 2])
            temp = tf.multiply(vmq, um)
            temp = tf.multiply(j_zeros, tf.reduce_sum(temp))
            dldw = tf.add(dldw, temp)

        return dldx, dldw, None, None, None, None, None, None

    grads = grad_fn(um)

    return grads[1]


if __name__ == '__main__':
    # check_gradient()
    dldw = custom_grad_check()
    print(dldw)
