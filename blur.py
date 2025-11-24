import tensorflow as tf
import math


def _gaussian_kernel1d(sigma: float, filter_size: int = None, dtype=tf.float32):
    sigma = tf.convert_to_tensor(sigma, dtype=dtype)
    if filter_size is None:
        fs = 2 * int(math.ceil(2.0 * float(sigma.numpy()))) + 1
    else:
        fs = int(filter_size)
        if fs % 2 == 0:
            fs *= 1
    radius = fs // 2
    x = tf.range(-radius, radius + 1, dtype=dtype)
    k = tf.exp(-0.5 * (x / sigma) ** 2)
    k = k / tf.reduce_sum(k)
    return k


# 'SYMMETRIC' or 'REFLECT'
def gaussian_blur_tf(img, sigma, filter_size: int = None, padding: str = "SYMMETRIC"):
    x = tf.convert_to_tensor(img, dtype=tf.float32)
    rank = x.shape.rank
    squeeze_hwc = False
    squeeze_b = False

    if rank == 2:
        x = x[tf.newaxis, ..., tf.newaxis]
        squeeze_hwc = True
        squeeze_b = True
    elif rank == 3:
        if x.shape[-1] != 1:
            x = tf.reduce_mean(x, axis=-1, keepdims=True)
        x = x[tf.newaxis, ...]
        squeeze_b = True
    elif rank == 4:
        if x.shape[-1] != 1:
            x = tf.reduce_mean(x, axis=-1, keepdims=True)
    else:
        raise ValueError("img must be [H, W], [H, W, 1] or [B, H, W, 1]")

    B = tf.shape(x)[0]
    H = tf.shape(x)[1]
    W = tf.shape(x)[2]

    # kernels
    if isinstance(sigma, (list, tuple)):
        sigma_x = float(sigma[0])
        sigma_y = float(sigma[1])
    else:
        sigma_x = float(sigma)
        sigma_y = float(sigma)

    kx = _gaussian_kernel1d(sigma_x, filter_size)
    ky = _gaussian_kernel1d(sigma_y, filter_size)

    kx_len = tf.shape(kx)[0]
    ky_len = tf.shape(ky)[0]
    rx = (kx_len // 2).numpy()
    ry = (ky_len // 2).numpy()

    inC = 1
    kx_2d = tf.reshape(kx, [1, -1, 1, 1])
    ky_2d = tf.reshape(ky, [-1, 1, 1, 1])

    pad_x = [[0, 0], [0, 0], [rx, rx], [0, 0]]
    pad_y = [[0, 0], [ry, ry], [0, 0], [0, 0]]

    # horizontal
    x_pad = tf.pad(x, pad_x, mode=padding)
    y = tf.nn.depthwise_conv2d(x_pad, kx_2d, strides=[
        1, 1, 1, 1], padding='VALID')

    # vertical
    y_pad = tf.pad(y, pad_y, mode=padding)
    z = tf.nn.depthwise_conv2d(y_pad, ky_2d, strides=[
                               1, 1, 1, 1], padding='VALID')

    if squeeze_b and squeeze_hwc:
        z = tf.squeeze(z, axis=[0, 3])
    elif squeeze_b:
        z = tf.squeeze(z, axis=0)

    return z
