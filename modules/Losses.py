"""Specialized loss functions.



Functions:
        complex_mean_squared_error: Complex Mean Square Error loss function

"""
import tensorflow as tf
import tensorflow.keras.backend as K

def complex_mean_squared_error(y_true, y_pred):
    """Complex Mean Square Error loss function.

    # Arguments
        y_true:#

        y_pred:

    """
    #y1 = y_true[0]
    #y2 = y_true[1]
    batch_size = tf.shape(y_pred)[0]
    sq = K.square(K.abs(y_pred - y_true))
    return K.mean(sq, axis=-1)
    # return K.mean(sq/K.cast(batch_size,sq.dtype), axis=-1)


def scaled_loss(scaling_factor):#
    def complex_mean_squared_error(y_true, y_pred):
        """Complex Mean Square Error loss function.
         # Arguments
            y_true:

            y_pred:
        """
        batch_size = tf.shape(y_pred)[0]
        sq = K.square(K.abs(y_pred - y_true))
        return K.mean(sq, axis=-1) / scaling_factor
         # return K.mean(sq/K.cast(batch_size,sq.dtype), axis=-1)
    return complex_mean_squared_error

def scaled_mse_loss(y_true, y_pred):
    import numpy as np
    y_t = y_true[:,0,:].reshape(y_pred.shape)
    scaling_factor = y_true[:,1,:]
    scaling_factor = np.real(scaling_factor[0,0])
    sq = np.square(np.abs(y_pred - y_t))
    return np.mean(sq, axis=-1) / scaling_factor

def mse_loss(y_true, y_pred):
    import numpy as np
    y_t = y_true[:,0,:].reshape(y_pred.shape)
    sq = np.square(np.abs(y_pred - y_t))
    return np.mean(sq, axis=-1)

def wrapper_scaled_loss(**split_args):
    def loss(y_true, y_pred):
        # split y_true into the actual y_true (=y_t) and the scaling factor
        y_t, scaling_factors = tf.split(y_true, **split_args)
        y_t = K.cast(y_t,y_pred.dtype)
        sq = K.square(K.abs(y_pred - y_t))
        # arbitrarily take the very first element of the array as actual
        # scaling factor
        scaling_factor = K.cast(scaling_factors[0, 0, 0], sq.dtype)
        # note that the scaling_factor is now multiplicative!
        #return K.mean(sq, axis=-1) / scaling_factor
        return K.mean(sq, axis=-1) / scaling_factor
    return loss