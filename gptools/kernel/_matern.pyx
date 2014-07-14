import cython
import numpy as np
from scipy.special import betainc
from libc.stdint cimport int32_t

cdef extern from "matern.h":
    double matern52(double *xi, double *xj,
                    int32_t* ni, int32_t* nj,
                    int32_t d, double* var) nogil


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef _matern52(double[:, ::1] Xi, double[:, ::1] Xj,
                int32_t[:, ::1] ni, int32_t[:, ::1] nj,
                double[::1] var):

    cdef int i, d, n
    n = Xi.shape[0]
    d = Xi.shape[1]
    if not (n == len(Xi) == len(Xj) == len(ni) == len(nj)):
        raise ValueError("Lengths don't match")
    if not (d == Xi.shape[1] == Xj.shape[1] == ni.shape[1] == nj.shape[1] == len(var)):
        raise ValueError("Widths don't match")

    cdef double[::1] out = np.zeros(n, dtype=np.float64)

    with nogil:
        for i in range(n):
            out[i] = matern52(&Xi[i, 0], &Xj[i, 0], &ni[i, 0], &nj[i, 0], d, &var[0])

    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef _betacdf_warpeed_matern52(double[:, ::1] Xi, double[:, ::1] Xj,
                                int32_t[:, ::1] ni, int32_t[:, ::1] nj,
                                double[::1] var, double[:] alpha,
                                double[:] beta):
    cdef int i, d, n
    n = Xi.shape[0]
    d = Xi.shape[1]
    if not (n == len(Xi) == len(Xj) == len(ni) == len(nj)):
        raise ValueError("Lengths don't match")
    if not (d == Xi.shape[1] == Xj.shape[1] == ni.shape[1] == \
            nj.shape[1] == len(var) == len(alpha) == len(beta)):
        raise ValueError("Widths don't match")
    if np.any(ni > 0) or np.any(nj > 0):
        raise NotImplementedError("Derivative observations not supported")

    cdef double[::1] out = np.zeros(n, dtype=np.float64)
    cdef double[:, ::1] wXi = betainc(alpha, beta, Xi)
    cdef double[:, ::1] wXj = betainc(alpha, beta, Xj)

    with nogil:
        for i in range(n):
            out[i] = matern52(&wXi[i, 0], &wXj[i, 0], &ni[i, 0], &nj[i, 0], d, &var[0])

    return out
