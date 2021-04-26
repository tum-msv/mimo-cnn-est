import numpy as np
from .Templates import Estimator_mimo_ML, Descriptor
from SCM3GPP.toeplitz_helpers import vec2mat, mat2vec
from training_CNN_mimo import pilot_matrix
from scipy.linalg import toeplitz

class ML(Estimator_mimo_ML, Descriptor):
    _object_counter = 1

    def __init__(self, snr, transform, name=None):
        self.snr = snr
        self.rho = None
        self.transform = transform

        if name is None:
            self.name = 'ML_' + transform.__name__ + '_' + str(ML._object_counter)
            ML._object_counter += 1
        else:
            self.name = name

    def valid(self, channel_config, snr, n_coherences, n_antennas_BS, n_antennas_MS, n_pilots):
        return self.snr == snr

    def estimate(self, y, n_pilots, n_antennas_MS,t_BS,t_MS):
        n_batches, n_coherence, mn = y.shape
        n_antennas_BS = int(mn / n_pilots)
        y_n = np.zeros([n_batches, n_coherence, n_antennas_BS * n_antennas_MS], dtype=complex)
        X = pilot_matrix(n_pilots, n_antennas_BS, n_antennas_MS)

        for i in range(n_batches):
            y_i = X.conj().T @ y[i, :, :].T
            y_n[i, :, :] = y_i.reshape([1, n_coherence, n_antennas_BS * n_antennas_MS])

        z = mat2vec(np.fft.fft2(vec2mat(y_n,n_antennas_BS,n_antennas_MS))) / np.sqrt(n_antennas_MS*n_antennas_BS)
        #z = self.transform(y_n,False)

        (n_batches, n_coherence, n_antennas) = z.shape  # input data in transformed domain
        if n_antennas_MS == n_pilots:
            Xprod = X @ X.conj().T
            x = np.real(Xprod[0,0])
            self.rho = (n_pilots * 10 ** (self.snr * 0.1)) / (x * n_antennas_MS)
            cest = np.sum(np.abs(z) ** 2, axis=1) - n_coherence / self.rho
            cest[cest < 0] = 0
            cest = np.reshape(cest, (n_batches, n_antennas))

            W = cest / (cest + n_coherence / self.rho)
        else:
            W = np.zeros((n_batches,n_antennas))
            for n_b in range(n_batches):
                Xprod = X.conj().T @ X
                C = np.kron( toeplitz(t_MS[n_b,:]) , toeplitz(t_BS[n_b,:]))
                sigma2 = np.real(np.trace(C @ Xprod) / (n_antennas_BS * n_pilots * 10 ** (self.snr * 0.1)))
                self.rho = (1 / sigma2)
                cest_batch = np.sum(np.abs(z[n_b,:,:]) ** 2, axis=0) - n_coherence / self.rho
                cest_batch[cest_batch < 0] = 0
                cest_batch = np.reshape(cest_batch, (1, n_antennas))
                W[n_b,:] = cest_batch / (cest_batch + n_coherence / self.rho)


        W = np.reshape(W, (n_batches, 1, n_antennas))

        hest = mat2vec(np.fft.ifft2(vec2mat(W * z,n_antennas_BS,n_antennas_MS))) * np.sqrt(n_antennas_BS*n_antennas_MS)
        return hest

    @property
    def description(self):
        return 'ML (59) O(M log M)'
