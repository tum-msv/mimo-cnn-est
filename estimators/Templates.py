class Estimator(object):
    def valid(self, channel_config, snr, n_coherences, n_antennas):
        return False

    def estimate(self, y, n_pilots):
        pass


class GenieEstimator(object):
    def valid(self, channel_config, snr, n_coherences, n_antennas):
        return False

    def estimate(self, h, t, y):
        pass


class Descriptor(object):

    @property
    def description(self):
        return ''

#MIMO case
class Estimator_mimo(object):
    def valid(self, channel_config, snr, n_coherences, n_antennas_BS, n_antennas_MS, n_pilots):
        return False

    def estimate(self, y, n_pilots, n_antennas_MS):
        pass

class Estimator_mimo_cnn(object):
    def valid(self, channel_config, snr, n_coherences, n_antennas_BS, n_antennas_MS, n_pilots):
        return False

    def estimate(self, y):
        pass

class Estimator_mimo_ML(object):
    def valid(self, channel_config, snr, n_coherences, n_antennas_BS, n_antennas_MS, n_pilots):
        return False

    def estimate(self, y, n_pilots, n_antennas_MS, t_BS, t_MS):
        pass

class GenieEstimator_mimo(object):
    def valid(self, channel_config, snr, n_coherences, n_antennas_BS, n_antennas_MS, n_pilots):
        return False

    def estimate(self, h, y):
        pass