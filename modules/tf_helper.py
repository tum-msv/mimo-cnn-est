import tensorflow as tf


def complexx(dtype):
    if dtype == tf.float32.name:
        return tf.complex64
    elif dtype == tf.float64.name:
        return tf.complex128
    else:
        TypeError('complexx(dtype): expects a floating point number of type float32 or float64 as input')
