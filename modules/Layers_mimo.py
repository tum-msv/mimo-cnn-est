"""Extra layers.

Function similar to the standard keras layer.
See description of individual layer for more information.

Layers:
        CirConv: Circular convolution layer.

        FFT: Fast Fourier Transform layer.

        IFFT: Inverse Fast Fourier Transform layer.

        MeanAbs2: Mean Absolute Squared layer.

"""
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras import activations, initializers, regularizers, constraints
from .tf_helper import complexx
from training_CNN_mimo import pilot_matrix


class CirConv(layers.Layer):
    """Circular convolution layer.

    This layer creates a convolution kernel of the same size as the last dimension of the input
    that is circular convolved with the layer input to produce a tensor of outputs.
    If `use_bias` is True, a bias vector is created and added to the outputs.
    Finally, if `activation` is not `None`,
    it is applied to the outputs as well.

    Input should be of shape (batches,1,channels) and is not tested for others.

    # Arguments
        output_tyoe: Specifies the type of the output tensor.

        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).

        use_bias: Boolean, whether the layer uses a bias vector.

        kernel_initializer: Initializer for the `kernel` weights matrix
            (see [initializers](../initializers.md)).

        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).

        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).

        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).

        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).

        kernel_constraint: Constraint function applied to the kernel matrix
            (see [constraints](../constraints.md)).

        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
    """
    def __init__(self,
                 output_type=None,
                 activation=None,
                 use_bias=False,
                 kernel_initializer='TruncatedNormal',
                 bias_initializer='uniform',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 n_antennas_BS=None,
                 n_antennas_MS=None,
                 **kwargs):

        super(CirConv, self).__init__(**kwargs)

        self.output_type = output_type
        self.activation = activations.get(activation)
        #self.activation = personal_activation()
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.n_antennas_BS = n_antennas_BS
        self.n_antennas_MS = n_antennas_MS

        self.kernel = None
        self.bias = None

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(shape=(1, 1, input_shape[-1]),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(input_shape[-1],),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        super(CirConv, self).build(input_shape)  # Be sure to call this at the end

    def call(self, input, **kwargs):

        #w = K.reverse(self.kernel, axes=-1)

        #n_filter_length = w.shape[-1]

        # arrange as 4D array for conv2d
        # input of the convolution
        #xx = K.reshape(input, (-1, n_filter_length, 1, 1))

        # repeat, and arrange as 4D-array for conv2d
        # kernel of the convolution
        #ww = K.reshape(K.stack((w, w)), (2 * n_filter_length, 1, 1, 1))

        #dims: n_eval_batches, n_filter_length
        #outputs = K.squeeze(K.squeeze(K.conv2d(xx, ww, strides=(1, 1), padding="same"), -1), -1)

        #outputs = K.expand_dims(outputs, axis=1)

    ################################################### 2d-convolution #################################################
        w = K.reverse(self.kernel, axes=-1)
        w1 = K.reshape(w, shape=(-1, w.shape[1], self.n_antennas_MS, self.n_antennas_BS))
        w1 = K.permute_dimensions(w1, pattern=(0, 1, 3, 2))

        n_filter_length = w1.shape[-2]
        n_filter_width = w1.shape[-1]
        n_filter_mult = n_filter_length * n_filter_width

        # arrange as 4D array for conv2d
        xx1 = K.reshape(input, (-1, n_filter_mult, 1, 1))
        xx1 = K.reshape(xx1, shape=(-1, self.n_antennas_MS, self.n_antennas_BS, 1))
        xx1 = K.permute_dimensions(xx1, pattern=(0, 2, 1, 3))

        # repeat, and arrange as 4D-array for conv2d
        wtemp = K.reshape(K.stack((w1,w1),axis = 2),(1,1,2*n_filter_length,-1))
        wtemp = K.reshape(K.stack((wtemp,wtemp),axis=3),(1,1,-1,2*n_filter_width))
        ww1 = K.permute_dimensions(wtemp, pattern=(2,3,0,1))

        # dims: n_eval_batches, n_filter_length
        outputs1 = K.conv2d(xx1, ww1, padding="same")
        outputs1 = K.permute_dimensions(outputs1, pattern=(0, 2, 1, 3))
        outputs1 = K.reshape(outputs1, shape=(-1,n_filter_mult,1))
        outputs = K.permute_dimensions(outputs1,pattern=(0,2,1))


        #outputs = K.squeeze(K.conv3d(xx1, ww, strides=(1, 1, 1), padding="same"), -1)
        #stop = 0

        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias)

        if self.activation is not None:
            outputs = self.activation(outputs)

        if self.output_type is not None:
            outputs = K.cast(outputs, self.output_type)

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(CirConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def get_kernal(self):
        kernel = self.kernel
        return K.batch_get_value(kernel)

    def get_bias(self):
        if not self.use_bias:
            return None
        bias = self.bias
        return K.batch_get_value(bias)

    def set_kernal(self, kernal_values):
        params = self.kernel
        if not params:
            return
        weight_value_tuples = []
        param_values = K.batch_get_value(params)
        if param_values.shape != kernal_values.shape:
            raise ValueError('Layer kernel shape ' +
                             str(param_values.shape) +
                             ' not compatible with '
                             'provided kernel shape ' + str(kernal_values.shape))
        weight_value_tuples.append((params, kernal_values))
        K.batch_set_value(weight_value_tuples)

    def set_bias(self, kernal_bias):
        if not self.use_bias:
            return
        params = self.bias
        if not params:
            return
        weight_value_tuples = []
        param_values = K.batch_get_value(params)
        if param_values.shape != kernal_bias.shape:
            raise ValueError('Layer bias shape ' +
                             str(param_values.shape) +
                             ' not compatible with '
                             'provided bias shape ' + str(kernal_bias.shape))
        weight_value_tuples.append((params, kernal_bias))
        K.batch_set_value(weight_value_tuples)


class FFT(layers.Lambda):
    """Fast Fourier Transform layer.

    This layer performes the FFT on the last dimension of the input tensor and presents
    the reult as the ouput tensor.

    If the input is not of a complex type it gets cast in to such one.

    The ouput is complex valued and normalized to 1/sqrt(input.shape[-1])
    """
    def __init__(self, n_pilots, n_antennas_BS, n_antennas_MS, **kwargs):
        if 'function' in kwargs:
            super(FFT, self).__init__(**kwargs)
        else:
            def func(a):
                import tensorflow as tf
                if a.dtype not in [tf.complex64, tf.complex128]:
                    a = tf.cast(a, dtype=complexx(K.floatx()))
                X = K.constant(pilot_matrix(n_pilots, n_antennas_BS, n_antennas_MS),dtype=complexx(K.floatx()))
                #a = tf.matmul(a , X)
                #X = tf.transpose(X, conjugate=True)
                X = tf.math.conj(X)
                a = tf.matmul(a, X)
                #a = tf.scan(lambda _, x: tf.matmul(x, X), a)
                #a = tf.matmul(X,a)
                #a = tf.reshape(a, [-1, tf.shape(a)[2]])
                #a = tf.matmul(a, X)
                #a = tf.reshape(a, [-1, tf.shape(a)[1], tf.shape(X)[1])

                a_shape = tf.shape(a)
                a = tf.reshape(a, shape=(-1, tf.shape(a)[1], n_antennas_MS, n_antennas_BS), name='a')
                a = tf.transpose(a, perm=[0,1,3,2])

                fft = tf.signal.fft2d(a) / tf.sqrt(tf.cast(a_shape[-1],a.dtype))

                fft = tf.transpose(fft,perm=[0,1,3,2])
                fft = tf.reshape(fft, shape=(-1,tf.shape(fft)[1],n_antennas_BS*n_antennas_MS))
                return fft

            super(FFT, self).__init__(func, **kwargs)


class IFFT(layers.Lambda):
    """Inverse Fast Fourier Transform layer.

    This layer performes the IFFT on the last dimension of the input tensor and presents
    the reult as the ouput tensor.

    If the input is not of a complex type it gets cast in to such one.

    The ouput is complex valued and normalized to *sqrt(input.shape[-1])
    """
    def __init__(self,n_pilots, n_antennas_BS, n_antennas_MS, **kwargs):
        if 'function' in kwargs:
            super(IFFT, self).__init__(**kwargs)
        else:
            def func(a):
                import tensorflow as tf
                if a.dtype not in [tf.complex64, tf.complex128]:
                    a = tf.cast(a, dtype=complexx(K.floatx()))

                a_shape = tf.shape(a)
                a = tf.reshape(a, shape=(-1, tf.shape(a)[1], n_antennas_MS, n_antennas_BS), name='a')
                a = tf.transpose(a, perm=[0,1,3,2])
                ifft = tf.signal.ifft2d(a) * tf.sqrt(tf.cast(a_shape[-1], a.dtype))
                ifft = tf.transpose(ifft,perm=[0,1,3,2])
                ifft = tf.reshape(ifft, shape=(-1,tf.shape(ifft)[1],n_antennas_BS*n_antennas_MS))
                return ifft

            super(IFFT, self).__init__(func, **kwargs)


class MeanAbs2(layers.Lambda):
    """Mean Absolute Squared layer.

    This layer takes the absolute value of the complex input, squares it and takes the mean in the 2 dimension.

    The input can be of real or complex type.

    The ouput shape is the same as the input.
    """
    def __init__(self, **kwargs):
        if 'function' in kwargs:
            super(MeanAbs2, self).__init__(**kwargs)
        else:
            def func(a):
                import tensorflow as tf
                return tf.cast(tf.reduce_mean(tf.square(tf.abs(a)), 1, keepdims=True), K.floatx())
            
            super(MeanAbs2, self).__init__(func, **kwargs)


class FFT2x(layers.Lambda):
    """Fast Fourier Transform layer.

    This layer performes the FFT on the last dimension of the input tensor and presents
    the reult as the ouput tensor.

    If the input is not of a complex type it gets cast in to such one.

    The ouput is complex valued and normalized to 1/sqrt(input.shape[-1])
    """
    def __init__(self, n_pilots, n_antennas_BS, n_antennas_MS, **kwargs):
        if 'function' in kwargs:
            super(FFT2x, self).__init__(**kwargs)
        else:
            def func(a):
                import tensorflow as tf
                #import numpy as np
                if a.dtype not in [tf.complex64, tf.complex128]:
                    a = tf.cast(a, dtype=complexx(K.floatx()))
                X = K.constant(pilot_matrix(n_pilots, n_antennas_BS, n_antennas_MS),dtype=complexx(K.floatx()))
                a = tf.matmul(a , X)
                a_shape = tf.shape(a)
                fft = tf.signal.fft(tf.concat([a, tf.zeros(a_shape, dtype=a.dtype)], -1)) / tf.sqrt(tf.cast(2 * a_shape[-1], dtype=a.dtype))

                return fft

            super(FFT2x, self).__init__(func, **kwargs)


class IFFT2x(layers.Lambda):
    """Inverse Fast Fourier Transform layer.

    This layer performes the IFFT on the last dimension of the input tensor and presents
    the reult as the ouput tensor.

    If the input is not of a complex type it gets cast in to such one.

    The ouput is complex valued and normalized to *sqrt(input.shape[-1])
    """
    def __init__(self, **kwargs):
        if 'function' in kwargs:
            super(IFFT2x, self).__init__(**kwargs)
        else:
            def func(a):
                import tensorflow as tf
                if a.dtype not in [tf.complex64, tf.complex128]:
                    a = tf.cast(a, dtype=complexx(K.floatx()))
                #X = K.constant(pilot_matrix(n_pilots, n_antennas),dtype=complexx(K.floatx()))
                a_shape = tf.shape(a)
                ifft = tf.split(tf.signal.ifft(a), 2, axis=-1)[0] * tf.sqrt(tf.cast(a_shape[-1], a.dtype))
                #ifft = ifft @ X
                #ifft = tf.matmul(ifft, tf.transpose(X))
                return ifft

            super(IFFT2x, self).__init__(func, **kwargs)
