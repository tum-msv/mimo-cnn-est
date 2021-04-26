import numpy as np
from .Templates import GenieEstimator_mimo, Descriptor
from training_CNN_mimo import pilot_matrix


class GenieOMP(GenieEstimator_mimo, Descriptor):
    _object_counter = 1

    def __init__(self, n_pilots, n_antennas_BS, n_antennas_MS, use_rate=False, name=None):
        self.use_rate = use_rate
        self.n_pilots = n_pilots
        self.n_antennas_BS = n_antennas_BS
        self.n_antennas_MS = n_antennas_MS

        if name is None:
            self.name = 'Genie_OMP_' + str(GenieOMP._object_counter)
            GenieOMP._object_counter += 1
        else:
            self.name = name

    def valid(self, channel_config, snr, n_coherences, n_antennas_BS, n_antennas_MS, n_pilots):
        return True

    def estimate(self, h, y):
        if self.use_rate:
            return self.omp_genie_rate(y, h)
        else:
            return self.omp_genie_mse(y, h)

    def omp_genie_mse(self, y, h, A=None):
        if A is None:
            #A = np.eye(h.shape[-1])
            A = pilot_matrix(self.n_pilots,self.n_antennas_BS,self.n_antennas_MS)
        n_batches, n_coherence, n_antennas = h.shape

        D = self._dft_matrix(self.n_antennas_BS, 4 * self.n_antennas_BS) # dictionary
        #Aeff = A.dot(D) # effective sensing matrix
        D_mimo = np.kron(np.eye(self.n_antennas_MS), D)
        Aeff = A @ D_mimo  # effective sensing matrix
        hest = np.zeros(h.shape, dtype=np.complex)

        for b in range(n_batches):
            evaluate_cost = lambda xhat: np.sum(np.abs(h[b, :, :].T - D_mimo.dot(xhat))**2)  # cost measured in "channel space"
            #hest[b, :,:] = D.dot(self._omp_genie(Aeff,y[b, :, :].T, evaluate_cost)[0]).T
            hest[b, :, :] = (D_mimo @ self._omp_genie(Aeff, y[b, :, :].T, evaluate_cost)[0]).T

        return hest

    def omp_genie_rate(self, y, h, A=None):
        if not A:
            A = np.eye(h.shape[-1])
        n_batches, n_coherence, n_antennas = h.shape

        D = self._dft_matrix(n_antennas, 4*n_antennas) # dictionary
        Aeff = A.dot(D)  # effective sensing matrix
        hest = np.zeros(h.shape, dtype=np.complex)

        for b in range(n_batches):
            # rate as cost
            evaluate_cost = lambda xhat: -np.sum([np.abs(h[b, i, :].conj().T.dot(D.dot(xhat[:, i])))**2
                                                  / max([1e-8,np.sum(np.abs(D.dot(xhat[:, i]))**2)])
                                                  for i in range(h.shape[1])])

            hest[b, :, :] = D.dot(self._omp_genie(Aeff,y[b, :, :].T, evaluate_cost)[0]).T

        return hest

    @staticmethod
    def _omp_genie(A, y , evaluate_cost,
                   k_max=100, B=None, find_supp=None, solve_restricted=None):
        if not B:
            B = A.conj().T

        if not find_supp:
            find_supp = lambda x: np.argmax(np.sum(np.abs(x)**2, axis=1))

        if not solve_restricted:
            solve_restricted = lambda L: np.linalg.lstsq(A[:, list(L)], y,rcond=None)[0]

        l = set()
        xtmp = np.zeros((A.shape[1],y.shape[1]), dtype=np.complex)
        xhat = np.zeros((A.shape[1],y.shape[1]), dtype=np.complex)

        cost = evaluate_cost(xhat)
        for k in range(k_max):
            cost_prev = cost
            # Union of support sets
            l = l.union([find_supp(B.dot(y - A.dot(xtmp)))])
            # Restricted LS estimate
            xtmp[list(l), :] = solve_restricted(l)

            # Calculate angle between y and ytmp
            cost = evaluate_cost(xtmp)

            if cost > cost_prev: # cost increases...
                break

            xhat[:] = xtmp[:]

        return xhat, l

    @staticmethod
    def _dft_matrix(n_antennas, n_grid):
        grid = np.linspace(-1, 1, n_grid + 1)[:n_grid]

        d = 1 / np.sqrt(n_antennas) * np.exp(1j * np.pi * np.outer(np.arange(n_antennas), grid.conj().T))
        return d

    @property
    def description(self):
        return 'Genie OMP O(M log M)'
