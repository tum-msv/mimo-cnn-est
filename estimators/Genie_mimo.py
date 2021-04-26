import numpy as np
from scipy.linalg import toeplitz
from .Templates import GenieEstimator, Descriptor
from training_CNN_mimo import pilot_matrix

class Genie(GenieEstimator, Descriptor):
    _object_counter = 1

    def __init__(self, snr, name=None):
        self.snr = snr
        self.rho = 10 ** (0.1 * snr)
        self.sigma2 = 1 / self.rho

        if name is None:
            self.name = 'Genie_Aided_' + str(Genie._object_counter)
            Genie._object_counter += 1
        else:
            self.name = name

    def valid(self, channel_config, snr, n_coherences, n_antennas_BS, n_antennas_MS, n_pilots):
        return self.snr == snr

    def estimate(self, h, t_BS, t_MS, y, n_antennas_MS):
        hest = np.zeros(h.shape, dtype=h.dtype)
        (n_batches, n_coherence, n_antennas) = h.shape
        mn = y.shape[-1]
        n_antennas_BS = int(n_antennas / n_antennas_MS)
        n_pilots = int(mn / n_antennas_BS)
        X = pilot_matrix(n_pilots, n_antennas_BS, n_antennas_MS)
        Xprod = X.conj().T @ X
        for b in range(n_batches):
            C_BS = toeplitz(t_BS[b, :])  # get full cov matrix#
            C_MS = toeplitz(t_MS[b, :])  # get full cov matrix
            C = np.kron(C_MS,C_BS)

            sigma2 = np.real(np.trace(C @ Xprod) / (n_antennas_BS * n_pilots * 10 ** (self.snr * 0.1)))
            noise_cov = np.identity(n_antennas_BS * n_pilots) * sigma2
            Cr = X @ C @ X.conj().T + noise_cov

            hest[b, :, :] = y[b, :, :].dot(np.linalg.inv(Cr).T).dot(X.conj()).dot(C.T)

        return hest

    @property
    def description(self):
        return 'Genie Aided'
