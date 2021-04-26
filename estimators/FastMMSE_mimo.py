import numpy as np
from .Templates import Estimator_mimo, Descriptor

from SCM3GPP.scm_helper_MIMO import scm_channel
from SCM3GPP.toeplitz_helpers import best_circulant_approximation, vec2mat, mat2vec, best_block_circulant_approximation
from training_CNN_mimo import pilot_matrix
from scipy.linalg import toeplitz, circulant

class FastMMSE(Estimator_mimo, Descriptor):
    _object_counter = 1

    def __init__(self, channel, snr, n_antennas_BS, n_antennas_MS, n_pilots, name=None):
        self.snr = snr
        self.n_pilots = n_pilots
        self.n_antennas_BS = n_antennas_BS
        self.n_antennas_MS = n_antennas_MS
        self.channel_config = channel.get_config()
        self.rho = []

        F_BS = np.fft.fft(np.eye(n_antennas_BS))
        F_MS = np.fft.fft(np.eye(n_antennas_MS))
        self.F = 1 / np.sqrt(n_antennas_BS * n_antennas_MS) * np.kron(F_MS,F_BS)

        _, t_BS, t_MS = scm_channel(np.array([0.0]), np.array([0.0]), np.array([1.0]), 1, n_antennas_BS,
                                         n_antennas_MS,sigma_BS=channel.path_sigma_BS,sigma_MS=channel.path_sigma_MS)
        C_BS = toeplitz(t_BS)
        C_MS = toeplitz(t_MS)
        C = np.kron(C_MS,C_BS)

        c_BS = best_circulant_approximation(t_BS)
        c_MS = best_circulant_approximation(t_MS)
        c = np.kron(c_MS, c_BS)

        self.X = pilot_matrix(n_pilots, n_antennas_BS, n_antennas_MS)
        Xprod = self.X.conj().T @ self.X
        sigma2 = np.real(np.trace(C @ Xprod) / (n_antennas_BS * n_pilots * 10 ** (self.snr * 0.1)))
        self.rho.append(1 / sigma2)

        self.w = c / ( c + 1 / self.rho[-1])  # MMSE filter at sample i
        self.v = mat2vec(np.fft.fft2(vec2mat(self.w,n_antennas_BS,n_antennas_MS)))
        self.bias = np.sum(np.log(np.abs(1 - self.w  )))

        if name is None:
            self.name = 'FE_' + str(FastMMSE._object_counter)
            FastMMSE._object_counter += 1
        else:
            self.name = name

    def valid(self, channel_config, snr, n_coherences, n_antennas_BS, n_antennas_MS, n_pilots):
        return self.channel_config == channel_config and self.snr == snr and self.n_antennas_BS == n_antennas_BS \
                    and self.n_antennas_MS == n_antennas_MS and self.n_pilots == n_pilots

    def estimate(self, y, n_pilots, n_antennas_MS):
        n_batches, n_coherence, mn = y.shape
        n_antennas_BS = int(mn / n_pilots)
        y_n = np.zeros([n_batches, n_coherence, n_antennas_BS*n_antennas_MS],dtype=complex)
        for i in range(n_batches):
            y_i = self.X.conj().T @ y[i, :, :].T
            y_n[i, :, :] = y_i.reshape([1, n_coherence, n_antennas_BS*n_antennas_MS])

        # calculate weights for all batches
        Y = vec2mat(y_n,n_antennas_BS,n_antennas_MS)
        Y = np.fft.fft2(Y) / np.sqrt(n_antennas_BS*n_antennas_MS)
        y = mat2vec(Y)

        cest = np.mean(np.abs(y)**2, axis=1)
        cest_fft = vec2mat(cest,n_antennas_BS,n_antennas_MS)
        cest_fft = np.fft.fft2(cest_fft)
        cest_fft = mat2vec(cest_fft)
        inner = np.conj(self.v) * cest_fft
        inner = mat2vec(np.fft.ifft2(vec2mat(inner,n_antennas_BS,n_antennas_MS)))
        exps = np.real(inner) *self.rho[-1]*n_coherence + self.bias * n_coherence
        exps = exps - np.tile(np.reshape(np.amax(exps, axis=-1),(n_batches, 1)),[1, n_antennas_BS*n_antennas_MS])

        weights = np.exp(exps)
        weights = weights / np.tile(np.reshape(np.sum(weights, axis=-1),(n_batches, 1)),[1, n_antennas_BS*n_antennas_MS])

        weights_fft = mat2vec(np.fft.fft2(vec2mat(weights,n_antennas_BS,n_antennas_MS)))
        f = np.real(mat2vec(np.fft.ifft2(vec2mat(self.v * weights_fft,n_antennas_BS,n_antennas_MS))))
        f = np.reshape(f, (-1, 1, n_antennas_BS*n_antennas_MS))

        x = y * f
        x = mat2vec(np.fft.ifft2(vec2mat(x,n_antennas_BS,n_antennas_MS))) * np.sqrt(n_antennas_BS*n_antennas_MS)
        return x

    @property
    def description(self):
        return 'FE O(M log M)'

