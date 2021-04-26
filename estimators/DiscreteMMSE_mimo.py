import numpy as np
from scipy.linalg import toeplitz
from .Templates import Estimator_mimo, Descriptor
from training_CNN_mimo import pilot_matrix

class DiscreteMMSE(Estimator_mimo, Descriptor):
    _object_counter = 1

    def __init__(self, channel, snr, n_antennas_BS, n_antennas_MS, n_samples, n_pilots, name=None):
        self.n_antennas_BS = n_antennas_BS
        self.n_antennas_MS = n_antennas_MS
        self.snr = snr
        self.n_pilots = n_pilots
        self.rho = 10 ** (0.1 * snr)
        self.W = list()
        self.bias = list()
        self.channel_config = channel.get_config()
        self.X = pilot_matrix(n_pilots, n_antennas_BS, n_antennas_MS)
        self.Xprod = self.X.conj().T @ self.X
        self.sigma2_all = np.zeros(n_samples)

        for i in range(n_samples):
            _, t_BS, t_MS = channel.generate_channel(1, 1, n_antennas_BS, n_antennas_MS)
            C_BS = toeplitz(t_BS)
            C_MS = toeplitz(t_MS)
            C = np.kron(C_MS, C_BS).T
            sigma2 = np.real(np.trace(C @ self.Xprod) / (n_antennas_BS * n_pilots * 10 ** (self.snr * 0.1)))
            self.sigma2_all[i] = sigma2
            self.W.append(C @ self.X.T.conj() @ (np.linalg.inv(self.X @ C @ self.X.T.conj() + sigma2 * np.eye(n_antennas_BS*n_pilots))))  # MMSE filter at sample i
            self.bias.append(np.real(np.linalg.slogdet(np.eye(n_antennas_BS*n_pilots) - self.X @ self.W[-1]))[1])  # bias term at sample i

        if name is None:
            self.name = 'GE_' + str(DiscreteMMSE._object_counter)
            DiscreteMMSE._object_counter += 1
        else:
            self.name = name

    def valid(self, channel_config, snr, n_coherences, n_antennas_BS, n_antennas_MS, n_pilots):
        return self.channel_config == channel_config and self.snr == snr and self.n_antennas_BS == n_antennas_BS \
               and self.n_antennas_MS == n_antennas_MS and self.n_pilots == n_pilots

    def estimate(self, y, n_pilots, n_antennas_MS):
        n_batches, n_coherence, n_prod = y.shape
        n_antennas_BS = int(n_prod / n_pilots)
        n_samples = len(self.W)

        # calculate weights for all batches
        exps = np.zeros((n_batches, n_samples))

        for b in range(n_batches):
            for t in range(n_coherence):
                for i in range(n_samples):
                    sigma2 = self.sigma2_all[i]
                    exps[b, i] += np.real(y[b, t, :] @ self.W[i].T @ self.X.T @ y[b, t, :].conj().T) * ( 1 / sigma2) + self.bias[i]

        exps = exps - np.tile(np.reshape(np.amax(exps, axis=-1),(n_batches, 1)),[1, n_samples])
        weights = np.exp(exps)
        weights = weights / np.tile(np.reshape(np.sum(weights, axis=-1),(n_batches, 1)),[1, n_samples])
        weights = weights.astype(self.W[0].dtype)

        # Calculate estimated MMSE filters and channel estimates
        hest = np.zeros((n_batches, n_coherence, n_antennas_BS * n_antennas_MS), dtype=np.complex)
        for b in range(n_batches):
            W_est = np.zeros((n_antennas_BS*n_antennas_MS,n_antennas_BS*n_pilots), dtype=self.W[0].dtype)
            for i in range(n_samples):
                W_est += self.W[i] * weights[b, i]
            hest[b, :, :] = y[b, :, :].dot(W_est.T)

        return hest

    @property
    def description(self):
        return 'GE O(M^3)'
