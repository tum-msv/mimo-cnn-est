from modules.Utils import crandn
import numpy as np
from scipy.linalg import toeplitz

def get_observation(h, t_BS, t_MS, snr, n_pilots, n_antennas_BS):
    n_antennas_mult = h.shape[-1]
    n_antennas_MS = int(n_antennas_mult / n_antennas_BS)
    n_batches = h.shape[0]
    n_coherence = h.shape[1]
    X = pilot_matrix(n_pilots,n_antennas_BS,n_antennas_MS)
    y = np.zeros([n_batches, n_coherence, n_pilots*n_antennas_BS], dtype=complex)
    Xprod = X.conj().T @ X
    fac = n_antennas_BS * n_pilots * 10 ** (snr / 10)
    #choose noise variance for each sample such that snr is fixed
    for n_b in range(n_batches):
        h_b = h[n_b, :, :].reshape([n_antennas_mult, n_coherence])
        prod = X @ h_b
        y[n_b, :, :] = prod.reshape([1, n_coherence, n_antennas_BS * n_pilots])
        C_BS = toeplitz(t_BS[n_b,:])
        C_MS = toeplitz(t_MS[n_b,:])
        C = np.kron(C_MS,C_BS)
        sigma = np.sqrt(np.real(np.trace(C @ Xprod) / fac))
        if n_pilots==1:
            y[n_b,:,:] = y[n_b,:,:] + sigma * crandn(n_coherence,n_antennas_BS)
        else:
            Sigma = np.zeros([n_coherence,n_pilots*n_antennas_BS],dtype=complex)
            for n_p in range(n_pilots):
                sig = sigma * crandn(n_coherence,n_antennas_BS)
                Sigma[:,n_p*n_antennas_BS:(n_p+1)*n_antennas_BS] = sig
            y[n_b,:,:] = y[n_b,:,:] + Sigma
    return y

def init_cnn_estimator_from_config(channel_config, snr, est_config, n_pilots, n_antennas_BS, n_antennas_MS, weights=None, History=None, parameters = None):
    import tensorflow.keras as K
    from modules import Losses
    from modules.MMSEEstimator_mimo import MMSEEstimator

    lr = None
    kernel_init = None
    kernel_reg = None
    # Initialize new Estimator
    est = MMSEEstimator(channel_config=channel_config, snr=snr, n_pilots=n_pilots, n_antennas_BS=n_antennas_BS,
                        n_antennas_MS=n_antennas_MS, name=est_config['name'])

    # Configuration of the circular convolution layer
    kernel_init = rand_exp(0.00001, 0.001)
    kernel_reg = rand_exp(0.0001, 0.1)
    circconv_args = dict(activation=est_config['activation'],
                         kernel_initializer=K.initializers.TruncatedNormal(0.,0.0001),
                         #kernel_initializer=K.initializers.TruncatedNormal(0., kernel_init),
                         kernel_regularizer=K.regularizers.l2(),
                         use_bias=True,
                         bias_initializer=K.initializers.constant(0.1),
                         n_antennas_MS= n_antennas_MS,
                         n_antennas_BS = n_antennas_BS)

    # Build and compile the estimator model
    est.build_model(shape=est_config['shape'],
                    number_of_circconv_layers=2,
                    circconv_args=circconv_args,
                    transform=est_config['transform'])

    est.model.compile(optimizer='adam',
                      loss=Losses.complex_mean_squared_error, metrics=[Losses.complex_mean_squared_error])
    #lr = rand_exp(0.00001,0.001)
    #lr = 0.0001

    #est.model.compile(optimizer=K.optimizers.Adam(learning_rate=lr),
    #                  loss=Losses.complex_mean_squared_error, metrics=[Losses.complex_mean_squared_error])
    #only for debugging:
    #est.model.run_eagerly = True
    #test = est.model.summary()
    if weights:
        est.model.set_weights(weights)

    parameters = dict()
    parameters['snr'] = snr
    parameters['kernel_init'] = kernel_init
    parameters['kernel_reg'] = kernel_reg
    parameters['learning_rate'] = lr
    return est, parameters


def training_worker_pilots(channel, snr, est_config, batch_size, n_pilots, n_antennas_BS, n_antennas_MS, pipe_connection, weights=None):
    """ worker methode for training estimators,

    every instatiation of worker performs one estimator rates

    :param channel: Test to complete
    :param snr: Test to complete
    :param est_config: Test to complete
    :param n_batches: Test to complete
    :param n_batche_size: Test to complete
    :param pbar_queue: Queue to communicate overall progress
    """
    import multiprocessing as mp
    import numpy as np
    from modules.Callbacks.MultitrainingCallback import MultitrainingCallback
    import time

    np.random.seed(mp.current_process().pid + int(time.time()))
    # Initialize new Estimator
    est, parameters = init_cnn_estimator_from_config(channel.get_config(), snr, est_config, n_pilots,
                                                     n_antennas_BS, n_antennas_MS, weights)

    def sample_generator(batch_size):
        """
        This methode returns a channel sample generator.

        :param batch_size: Batch size
        :return: Generator for channel samples
        """
        def gen(shape):
            while True:
                h, t_BS, t_MS = channel.generate_channel(batch_size, *shape)

                y = get_observation(h, t_BS, t_MS, snr, n_pilots, n_antennas_BS)
                yield y, h
        return gen

    # validation set
    #val_x = []
    #val_y = []
    #generator = sample_generator(batch_size)
    #for val_iter in range(5):
    #    data_x, data_y = next(generator((1,n_antennas_BS, n_antennas_MS)))
    #    if list(val_x):
    #        val_x = np.concatenate((val_x, data_x), axis=0)
    #        val_y = np.concatenate((val_y, data_y), axis=0)
    #    else:
    #        val_x = data_x.copy()
    #        val_y = data_y.copy()
    #validation_set = val_x, val_y


    # Fit Model
    generator = sample_generator(batch_size)
    History = est.model.fit(x = generator((1, n_antennas_BS, n_antennas_MS)),
                                            epochs=est_config['epochs'],
                                            steps_per_epoch=est_config['steps_per_epoch'],
                                            verbose=0,
                                            callbacks=[MultitrainingCallback(pipe_connection)],
                                            #validation_data=validation_set,
                                            #validation_steps=validation_steps
                            )

    est_config['shape'] = est.shape
    return channel.get_config(), snr, est_config, est.model.get_weights(), History.history, parameters


def eval_worker_pilots(channel, snr, estimators, n_batche_size, n_coherences, n_antennas_BS, n_antennas_MS, n_pilots,
                       pipe_connection, plot_axis):
    import multiprocessing as mp
    import numpy as np
    import time
    from estimators import Templates
    from itertools import product
    #from modules.Utils import crandn

    # initialize CNN estimators and extract the correct estimators
    inter_est = []
    for est in estimators:
        if isinstance(est, tuple):
            est = list(est)
            est[2]['shape'] = (n_coherences, int(n_pilots * n_antennas_BS))
            #est.insert(3,n_pilots)
            #est.insert(4, n_antennas_BS)
            #est.insert(5, n_antennas_MS)
            est = tuple(est)
            est, _ = init_cnn_estimator_from_config(*est)

        if est.valid(channel.get_config(), snr, n_coherences, n_antennas_BS, n_antennas_MS, n_pilots):
            inter_est.append(est)

    estimators = inter_est

    process_name = mp.current_process().name
    pipe_connection.send([process_name + '_eval', dict(total=len(estimators), leave=False)])

    np.random.seed(mp.current_process().pid + int(time.process_time()))

    rho = 10 ** (0.1 * snr)

    h, t_BS, t_MS = channel.generate_channel(n_batche_size, n_coherences, n_antennas_BS, n_antennas_MS)
    y = get_observation(h, t_BS, t_MS, snr, n_pilots, n_antennas_BS)

    results = list()
    for est in estimators:
        if issubclass(est.__class__, Templates.Estimator_mimo_cnn): #CNN estimator
            hest = est.estimate(y)
        if issubclass(est.__class__, Templates.Estimator_mimo): # GE / SE / FE
            hest = est.estimate(y, n_pilots, n_antennas_MS)
        if issubclass(est.__class__, Templates.Estimator_mimo_ML): #ML-estimator
            hest = est.estimate(y, n_pilots, n_antennas_MS, t_BS, t_MS)
        if issubclass(est.__class__, Templates.GenieEstimator_mimo): #Genie-OMP
            hest = est.estimate(h, y)
        elif issubclass(est.__class__, Templates.GenieEstimator): #Genie / LS estimator
            #Genie-mmse
            hest = est.estimate(h, t_BS, t_MS, y, n_antennas_MS)
            #LS solution
            #hest = np.zeros(h.shape,dtype=complex)
            #X = pilot_matrix(n_pilots, n_antennas_BS, n_antennas_MS)
            #for n_b in range(n_batche_size):
            #    hest_T = np.linalg.lstsq(X,y[n_b,:,:].T,rcond=None)
            #    hest[n_b,:,:] = hest_T[0].T
        else:
            TypeError('{} has to conform to {} or {}'.format(est.__class__.__name__,
                                                             Templates.Estimator.__name__,
                                                             Templates.GenieEstimator.__name__))

        mse = np.sum(np.abs(h - hest) ** 2) / h.size
        rate = 0.0
        normalisation_factor = h[:, :, 1].size
        for batch, coherenc in product(range(n_batche_size), range(n_coherences)):
            rate += np.log2(1 + rho * np.abs(np.vdot(h[batch, coherenc, :], hest[batch, coherenc, :])) ** 2 /
                            np.maximum(1e-8, np.sum(np.abs(hest[batch, coherenc, :]) ** 2)))

        rate = rate / normalisation_factor

        if plot_axis == "antennas":
            arg = n_antennas_BS
        elif plot_axis == "snr":
            arg = snr
        elif plot_axis == "coherences":
            arg = n_coherences
        else:
            arg = None
            print("Error in eval_worker; plot_axis not defined")
        results.append(dict(id=id_func(est.name, arg),
                            mse=mse, rate=rate))
        pipe_connection.send([process_name + '_eval', dict(update=1)])

    pipe_connection.send([process_name + '_eval', 'top_level'])
    return results

def id_func(name, arg):
    return ('_'.join(str.split(name, '_')[:-1]), arg)

def pilot_matrix(n_pilots, n_antennas_BS, n_antennas_MS):
    if n_pilots >= n_antennas_MS:
        F = np.fft.fft(np.eye(n_pilots))
        x = np.sqrt(1/(n_antennas_MS)) * F[:n_antennas_MS, :]
    else:
        F = np.fft.fft(np.eye(n_antennas_MS))
        x = np.sqrt(1/(n_antennas_MS)) * F[:,:n_pilots]
    X = np.kron(x.T,np.eye(n_antennas_BS))
    return X

def rand_exp(left, right):
    return np.exp(np.log(left) + np.random.rand()*(np.log(right) - np.log(left)))

def rand_geom(left, right):
    return np.round(rand_exp(left, right)).astype('int')