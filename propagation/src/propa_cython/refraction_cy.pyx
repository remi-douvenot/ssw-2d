##
# @brief Cython function that applies half a phase screen before or after propagation
# @author R. Douvenot
# @package apply_refractive_index
# @date 10/09/21
# @version OK
#
# @details Function that applies half a phase screen before or after propagation.
# def apply_refractive_index_cy(u_x, n_index, config):
#
# @params[in] u_x : reduced electric field (complex array)
# @params[in] n_index : phase screen (real array)
# @params[in] config : class with the parameters
# @params[in] u_x : reduced electric field (complex array)
##

import scipy.constants as cst
import numpy as np
cimport numpy as np
from cython cimport boundscheck, wraparound, cdivision, nonecheck
cdef extern from "complex.h":  # import complex number library
    double complex exp(double complex z)
    pass
@boundscheck(False)
@wraparound(False)
@cdivision(True)
@nonecheck(False)

def apply_refractive_index_cy(double complex[:] u_x, const double[:] n_index, const double freq, const double x_step):

    cdef Py_ssize_t ii
    cdef double c0 = cst.c
    cdef double pi = np.pi
    cdef double complex phi
    cdef double k0 = 2*pi*freq / c0
    cdef int n_u = u_x.shape[0]
    # apply the phase screen of one step delta_x
    # half the refraction applied before and after propagation

    for ii in range(0, n_u):
        u_x[ii] = u_x[ii] * np.exp(-1j * k0 * (n_index[ii]-1)/2 * x_step)
    return u_x