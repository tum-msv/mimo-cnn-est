import types
from itertools import product

import numpy as np
from scipy.interpolate import interp1d
import tensorflow.keras as K
from . import Layers_mimo
from .tf_helper import complexx
from estimators.Templates import Estimator_mimo_cnn, Descriptor


class MMSEEstimator(Estimator_mimo_cnn, Descriptor):
    """MMSE Estimator.

    This class provides a wrapper for the neural network used for estimating the channel matrix.
    """

    _object_counter = 0

    def __init__(self,
                 channel_config,
                 snr,
                 n_pilots,
                 n_antennas_BS,
                 n_antennas_MS,
                 name=None,
                 verbose=0):
        """Initialise an MMSEEstimator object.

        :param name: A name for the MMSEEstimator keras model.
                     Set the default name to 'MMSEEstimator' if the user does not
                     give a name for the MMSEEstimator.
        :param verbose: Select whether or not to display hints
                        and the model summary.
        """
        self.channel_config = channel_config
        self.snr = snr
        self.n_pilots = n_pilots
        self.n_antennas_BS = n_antennas_BS
        self.n_antennas_MS = n_antennas_MS
        self.pilot_args = dict()

        self.shape = None
        self.transform = None
        self.number_of_circconv_layers = None
        self.transform_args = None
        self.circconv_args = None

        self.model = None
        self.verbose = verbose

        if name is None:
            self.name = 'mmse_' + str(MMSEEstimator._object_counter)
            MMSEEstimator._object_counter += 1
        else:
            self.name = name


    def build_model(self,
                    shape,
                    transform='fft',
                    number_of_circconv_layers=2,
                    transform_args=None,
                    circconv_args=None):
        """Build a Keras Tensorflow Model.

        Function to build an Keras model with the parameters passed by the user.

        :param shape: Defines the input shape of the Keras model.
        :param transform: Describes what type of input/ouput transform should be used.
                          Currently provided transfroms:
                          'fft': fft at input, ifft at output
        :param number_of_circconv_layers: How many layers of circular convolutions should be used.
                                          Remarke that the last layer always uses a linear output activation.
        :param transform_args: Provide transform layer specific arguments.
        :param circconv_args: Provide circular convolution layer specific arguments.
        """

        self.shape = shape
        self.transform = transform
        self.transform_args = transform_args or dict()
        self.circconv_args = circconv_args or dict()
        self.number_of_circconv_layers = number_of_circconv_layers

        # Create the estimator input
        input_layer = K.Input(dtype=complexx(K.backend.floatx()), shape=self.shape)

        # Get input and output layer
        input_transform_layer, ouput_transform_layer = self.set_transform_layers(self.transform, self.transform_args,
                                                                 self.n_antennas_BS, self.n_antennas_MS, self.n_pilots)

        # Set input layer
        a = input_transform_layer(input_layer)

        b = Layers_mimo.MeanAbs2()(a)

        # Build the circular convolution layer except the last
        for _ in range(self.number_of_circconv_layers-1):
            b = Layers_mimo.CirConv(**self.circconv_args)(b)

        # Last circular convolution layer doesn't have a activation function
        if 'activation' in circconv_args:
            circconv_args.pop('activation')

        b = Layers_mimo.CirConv(output_type=complexx(K.backend.floatx()),
                           **self.circconv_args)(b)

        c = K.layers.multiply([a, b])

        # Set output layer
        output_layer = ouput_transform_layer(c)

        # Set keras model
        self.model = K.Model(inputs=input_layer, outputs=output_layer, name=self.name)

        if self.verbose == 1:
            self.model.summary()


    @staticmethod
    def set_transform_layers(transform, transform_args, n_antennas_BS, n_antennas_MS, n_pilots):
        """Set the transformation layers

        This function returns the input and output transformation layer.

        :param transform: Describes what type of input/ouput transform should be used.
                          Currently provided transfroms:
                          'fft': fft at input, ifft at output,
                          'fft2x': 2MxM fft at input, 2MxM ifft at output,
                          'user_defined': user defined intput and output layer.
        :param transform_args: Provide transform layer specific arguments.
                               'user_defined': Has to contain the 'input_layer' and 'output_layer'
        :return: input and output transform layer as tuple
        """
        layer_switcher = {
            'fft': (Layers_mimo.FFT, Layers_mimo.IFFT),
            'fft2x': (Layers_mimo.FFT2x, Layers_mimo.IFFT2x),
            'user_defined': (None, None)
        }

        if transform not in layer_switcher.keys():
            raise ValueError(
                'Input/Output tranform type {} not supported.'
                'Supported values are: {}.'.format(transform, layer_switcher.keys()))

        input_transform_layer, ouput_transform_layer = layer_switcher[transform]
        if transform == 'fft':
            input_transform_layer = input_transform_layer(n_pilots=n_pilots, n_antennas_BS=n_antennas_BS, n_antennas_MS=n_antennas_MS, **transform_args)
            ouput_transform_layer = ouput_transform_layer(n_pilots=n_pilots, n_antennas_BS=n_antennas_BS, n_antennas_MS=n_antennas_MS, **transform_args)
        elif transform == 'fft2x':
            input_transform_layer = input_transform_layer(n_pilots=n_pilots, n_antennas_BS=n_antennas_BS, n_antennas_MS=n_antennas_MS, **transform_args)
            ouput_transform_layer = ouput_transform_layer(**transform_args)
        elif transform == 'user_defined':
            if 'input_layer' not in transform_args or 'output_layer' not in transform_args:
                ValueError("For the 'user_defined' option transfrom_args has to include the input and output transform"
                           "layer as 'input_layer' and 'output_layer'")

            if not issubclass(transform_args['input_layer'], K.layers.Layer) or not issubclass(
                    transform_args['output_layer'], K.layers.Layer):
                TypeError("Input and output layer have to be subclass of Keras.layers.Layer")

            input_transform_layer = transform_args['input_layer']
            ouput_transform_layer = transform_args['output_layer']

        return input_transform_layer, ouput_transform_layer


    def evaluate(self, generator, n_batches, snr):
        mse = 0.0
        result = 0.0
        rho = 10 ** (0.1 * snr)
        for _ in range(n_batches):

            y, h = next(generator(self.shape))
            h_pred = self.model.predict(y,
                                        batch_size=y.shape[0],
                                        verbose=False)

            # mse += np.sum(np.square(np.abs(h - h_pred))) / h.size / n_eval_batches

            _, mse_eval = self.model.evaluate(y, h,
                                              batch_size=y.shape[0],
                                              verbose=False)
            mse += mse_eval / n_batches

            for b, t in product(range(h_pred.shape[0]), range(h_pred.shape[1])):
                if np.sum(np.square(np.abs(h_pred[b, t, :]))) < 1e-8:
                    continue

                a = np.log2(1 + rho * np.square(np.abs(np.dot(h[b, t, :], h_pred[b, t, :])))
                            / np.maximum(1e-8, np.sum(np.square(np.abs(h_pred[b, t, :])))))
                result += a / h_pred[:, :, 1].size / n_batches
        return mse, result


    @property
    def is_compiled(self):
        return hasattr(self.model, 'train_function')


    def valid(self, channel_config, snr, n_coherences, n_antennas_BS, n_antennas_MS, n_pilots):
        return self.channel_config == channel_config and self.snr == snr and \
               self.shape == (n_coherences, int(n_pilots * n_antennas_BS)) and \
               self.n_antennas_MS==n_antennas_MS and self.n_pilots == n_pilots


    def estimate(self, y):
        y=y.astype('complex64')
        h_pred = self.model.predict(y,
                                    batch_size=y.shape[0],
                                    verbose=False)
        return h_pred


    @property
    def description(self):
        return 'CNN'


    def __getstate__(self):
        return self.__dict__


    def __setstate__(self, state):
        self.__dict__ = state