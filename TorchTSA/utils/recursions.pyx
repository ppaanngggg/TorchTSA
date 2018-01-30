cimport numpy as np
cimport cython


ctypedef np.double_t DTYPE_t

@cython.wraparound(False)
@cython.boundscheck(False)
def arma_recursion(
    np.ndarray[DTYPE_t, ndim=1] _latent,
    np.ndarray[DTYPE_t, ndim=1] _theta,
):
    cdef int latent_num = _latent.shape[0]
    cdef int theta_num = _theta.shape[0]
    cdef int i, j
    cdef double total

    for i in range(theta_num, latent_num):
        total = 0.0
        for j in range(theta_num):
            total += _latent[i - j - 1] * _theta[j]
        _latent[i] -= total

@cython.wraparound(False)
@cython.boundscheck(False)
def garch_recursion(
    np.ndarray[DTYPE_t, ndim=1] _latent,
    np.ndarray[DTYPE_t, ndim=1] _beta,
):
    cdef int latent_num = _latent.shape[0]
    cdef int beta_num = _beta.shape[0]
    cdef int i, j
    cdef double total

    for i in range(beta_num, latent_num):
        total = 0.0
        for j in range(beta_num):
            total += _latent[i -j - 1] * _beta[j]
        _latent[i] += total