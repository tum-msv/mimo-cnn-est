import numpy as np

def fft(x, transpose=False, X=None):
    if not transpose:
        result = np.fft.fft(x) / np.sqrt(x.shape[-1])
    else:
        result = np.fft.ifft(x) * np.sqrt(x.shape[-1])
    return result


def fft2x(x, transpose=False, X=None):
    if not transpose:
        result = np.fft.fft(np.concatenate([x, np.zeros(x.shape, dtype=x.dtype)], axis=-1)) / np.sqrt(2 * x.shape[-1])
    else:
        result = np.split(np.fft.ifft(x), 2, axis=-1)[0] * np.sqrt(x.shape[-1])
    return result


def fft_2d(x, first_dim, sec_dim, transpose=False):
    #reshape to matrix form to apply 2D-DFT
    x_matrix = np.zeros([x.shape[0],x.shape[1],sec_dim,first_dim],dtype=complex)
    for i in range(x.shape[0]):
        x_matrix[i,:,:,:] = x[i,:,:].reshape([x.shape[1],sec_dim,first_dim],order='F')

    #apply 2D-DFT
    if not transpose:
        x_matrix = np.fft.fft2(x_matrix) / np.sqrt(x.shape[-1])
    else:
        x_matrix = np.fft.ifft2(x_matrix) * np.sqrt(x.shape[-1])

    #reshape to vector again
    result = np.zeros(x.shape,dtype=complex)
    for i in range(x.shape[0]):
        result[i,:,:] = x_matrix[i,:,:,:].reshape([x.shape[1],x.shape[2]],order='F')

    return result