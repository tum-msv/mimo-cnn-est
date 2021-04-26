import numpy as np
from scipy.linalg import toeplitz
from .Templates import Estimator_mimo, Descriptor
from training_CNN_mimo import pilot_matrix
from estimators.Transforms import fft_2d

class StructuredMMSE(Estimator_mimo, Descriptor):
    _object_counter = 1

    def __init__(self, channel, snr, n_antennas_BS, n_antennas_MS, transform, n_samples, n_pilots, name=None):
        self.n_antennas_BS = n_antennas_BS
        self.n_antennas_MS = n_antennas_MS
        self.snr = snr
        self.n_pilots = n_pilots
        self.rho = 10 ** (0.1 * snr)
        self.transform = transform
        self.channel_config = channel.get_config()
        self.sigma2_all = np.zeros(n_samples)

        X = pilot_matrix(n_pilots,n_antennas_BS,n_antennas_MS)
        Xprod = X.conj().T @ X
        Q_BS = transform(np.eye(n_antennas_BS), False)
        Q_MS = transform(np.eye(n_antennas_MS), False)
        Q = np.kron(Q_MS,Q_BS)

        Q = Q.T @ X.conj().T
        A = np.linalg.pinv(np.abs(Q.dot(Q.conj().T))**2)

        self.W = np.zeros((n_samples, Q.shape[0]))
        self.W_scaled = np.zeros(self.W.shape)
        self.bias = np.zeros(n_samples)

        for i in range(n_samples):
            _, t_BS, t_MS = channel.generate_channel(1, 1, n_antennas_BS, n_antennas_MS)
            C_BS = toeplitz(t_BS)
            C_MS = toeplitz(t_MS)
            C = np.kron(C_MS,C_BS)
            sigma2 = np.real(np.trace(C @ Xprod) / (n_antennas_BS * n_pilots * 10 ** (self.snr * 0.1)))
            self.sigma2_all[i] = sigma2
            W = C @ X.conj().T @ np.linalg.pinv(X @ C @ X.conj().T + sigma2 * np.eye(n_antennas_BS*n_pilots))

            self.W[i, :] = A.dot(np.real(np.diag(Q @ X @ W @ Q.conj().T,0)))
            self.bias[i] = np.real(np.linalg.slogdet(np.eye(n_antennas_BS*n_pilots) -
                                                     Q.conj().T.dot(np.diag(self.W[i, :])).dot(Q))[1])
            #multiply directly with sample-based noise-cov
            self.W_scaled[i,:] = self.W[i,:] * (1 / sigma2)

        if name is None:
            self.name = 'SE_' + transform.__name__ + '_' + str(StructuredMMSE._object_counter)
            StructuredMMSE._object_counter += 1
        else:
            self.name = name

    def valid(self, channel_config, snr, n_coherences, n_antennas_BS, n_antennas_MS, n_pilots):
        return self.channel_config == channel_config and self.snr == snr and self.n_antennas_BS == n_antennas_BS \
               and self.n_antennas_MS == n_antennas_MS and self.n_pilots == n_pilots

    def estimate(self, y, n_pilots, n_antennas_MS):
        (n_batches,n_coherences,mn) = y.shape
        n_antennas_BS = int(mn/n_pilots)
        X = pilot_matrix(n_pilots,n_antennas_BS,n_antennas_MS)
        y_n = np.zeros([n_batches,n_coherences,n_antennas_BS*n_antennas_MS],dtype=complex)
        for i in range(n_batches):
            y_i = X.conj().T @ y[i,:,:].T #reshape([n_antennas*n_pilots,n_coherences])
            y_n[i,:,:] = y_i.reshape([1,n_coherences,n_antennas_BS*n_antennas_MS])
        z = fft_2d(y_n, first_dim=n_antennas_MS, sec_dim=n_antennas_BS, transpose=False)
        (n_batches, n_coherence, n_antennas) = z.shape
        n_samples = self.W.shape[0]

        cest = np.sum(np.abs(z)**2, axis=1)
        cest = np.reshape(cest, (n_batches, n_antennas))

        exps = cest.dot(self.W_scaled.T) + self.bias * n_coherence # note cest sum not mean
        exps = exps - np.tile(np.reshape(np.amax(exps, axis=-1),(n_batches, 1)),[1, n_samples])

        weights = np.exp(exps)
        weights = weights / np.tile(np.reshape(np.sum(weights, axis=-1),(n_batches, 1)),[1, n_samples])

        F = weights.dot(self.W.conj())
        F = np.reshape(F, (n_batches, 1, n_antennas_BS * n_antennas_MS))

        hest = fft_2d(F * z, first_dim=n_antennas_MS, sec_dim=n_antennas_BS, transpose=True)
        return hest

    @property
    def description(self):
        transfrom_name_switcher = {
            'fft': 'Circular',
            'fft2x': 'Toeplitz'
        }

        if self.transform.__name__ in transfrom_name_switcher.keys():
            return transfrom_name_switcher[self.transform.__name__] + ' SE O(M^2)'
        else:
            return self.transform.__name__ + ' SE O(M^2)'
