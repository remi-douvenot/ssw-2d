# This file is part of SSW-2D.
# SSW-2D is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# SSW-2D is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
# of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Foobar. If not, see
# <https://www.gnu.org/licenses/>.

##
# @package: wavelet_propag_one_step2
# @author: Remi Douvenot
# @date: 15/03/2023
# @version: WIP
#
# @brief One step of the SSW 2D free-space propagation -- Cython version
# @param[in] wv_x Wavelet decomposition of the field before the free-space propagation
# @param[in] dictionary Pre-generated wavelet propagators.
# @param[in] family Str wavelet family
# @param[in] wv_ll Integer max wavelet level decomposition
# @param[in] n_z Integer total size of the signal (including image layer and apodisation)
# @param[out] wv_x_dx Wavelet decomposition of the field after the free-space propagation
# @details Cython package that applies the free-space propagators to all the nonzero wavelet coefficients
#
# wv_x structure is [A(J) D(J) D(J-1) ..... D(1)]
# where A(J) is the approximation coefficient vector at the Jth level while D(n) are the detail coefficient vectors
# at the nth level.
#
# Compilation:
# >> cython -3 -a wavelet_propag_one_step.pyx
# >> gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -ffast-math -I/usr/include/python3.10 -o wavelet_propag_one_step.so wavelet_propag_one_step.c
##

# import time
from src.wavelets_cython.wavelets_operations import normalized_indices, calculate_dilation
import numpy as np
cimport numpy as np
from cython cimport boundscheck, wraparound, cdivision, nonecheck

cdef extern from "complex.h":  # import complex number library
    pass

@boundscheck(False)
@wraparound(False)
@cdivision(True)
@nonecheck(False)
##
# @package: wavelet_propag_one_step_cy
# @author: Remi Douvenot
# @date: 15/03/2023
# @version: WIP
#
# @brief One step of the SSW 2D free-space propagation -- Cython version
# @param[in] wv_x Wavelet decomposition of the field before the free-space propagation
# @param[in] dictionary Pre-generated wavelet propagators.
# @param[in] family Str wavelet family
# @param[in] wv_ll Integer max wavelet level decomposition
# @param[in] n_z Integer total size of the signal (including image layer and apodisation)
# @param[out] wv_x_dx Wavelet decomposition of the field after the free-space propagation
# @details Cython package that applies the free-space propagators to all the nonzero wavelet coefficients
#
# wv_x structure is [A(J) D(J) D(J-1) ..... D(1)]
# where A(J) is the approximation coefficient vector at the Jth level while D(n) are the detail coefficient vectors
# dictionary structure is [p(0) p(1) p(n) ..... p(2**L)]
# where p(n) is the nth propagator (wavelets) put in the same shape as wv_x
##

def wavelet_propag_one_step_cy(const int n_z, const double complex[:] wv_x, double complex[:] dictionary, const int wv_ll,
                               const int[:] n_start_propa, const double v_p):

    # --- Init the propagated wavelet coefficient --- #
    # INPUTS (constants, defined in the function call)
    # OUTPUT: vector full of zeros, same structure as wv_x: wavelet decomposition after propagation
    wv_x_dx = np.zeros(n_z, dtype = np.complex128)
    cdef double complex[:] wv_x_dx_cy = wv_x_dx

    # -------- START --------- #
    # --- Define variables --- #
    # ------------------------ #

    # current wavelet coefficient in the loop
    cdef double complex wv_coef, propa_coef
    # number of coefficients on the highest levels (scaling function) of the SIGNAL vectors (wv_x and wv_x_dx)
    cdef int n_scal = int(n_z / (2 ** wv_ll))
    # number of points with respect to the scaling function at each level
    cdef int[:] list_q = calculate_dilation(wv_ll)
    # index of the coefficients at each level normalized with respect to the scaling function size at each level
    cdef int[:] norm_indices = normalized_indices(wv_ll)
    # indices in the loops
    cdef Py_ssize_t ii_lvl_wv, ii_lvl_pr, ii_ker  # loops on the levels
    cdef Py_ssize_t ii_wv, ii_ind, ii_propa  # loops on the coefficients
    cdef Py_ssize_t[:] nonzero
    # integers
    cdef int q_wv, q_pro, ii_q  # levels of dilation and associate counter
    cdef Py_ssize_t n_start_wv_lvl, n_stop_wv_lvl  # where index of level of wv_x begins / ends
    cdef Py_ssize_t n_start_propa_lvl, n_stop_propa_lvl  # where index of propagators of current level begins / ends in "dictionary"
    cdef Py_ssize_t n_start_propa_lvl_q, n_stop_propa_lvl_q  # where index of current propagator begins / ends in "dictionary"
    cdef Py_ssize_t n_start_propa_lvl_q_lvl, n_stop_propa_lvl_q_lvl  # the same with the current level of the propagator
    cdef Py_ssize_t n_start_wvdx_lvl, n_stop_wvdx_lvl  # in the same level, min/max indices of the level in wv_x_dx
    cdef Py_ssize_t ii_wv_dilate  # current position modified by dilations/contractions
    cdef Py_ssize_t n_coef  # number of coefficients at propagated level for wv_x_dx
    cdef Py_ssize_t n_propa, n_scal_pr  # size of one propagator, of one scaling function
    cdef Py_ssize_t n_ker, n_ker2  # size and center of the current level of the propagator
    cdef Py_ssize_t ii, ind_max, ind_min
    # ------------------------ #
    # --- Define variables --- #
    # --------- END ---------- #

    # ----------------------------------------------------------- #
    # ------------- Propagation in the wavelet domain ----------- #
    # --- LOOP ON THE WAVELET LEVELS OF THE INPUT VECTOR WV_X --- #
    # ----------------------------------------------------------- #

    # TODO: calculate all the stop and start coefficients just once in a dedicated loop.
    # --- LOOP ON THE LEVELS OF W_X --- #
    for ii_lvl_wv in range(0, wv_ll + 1):
        # number of propagators at this level
        q_wv = list_q[ii_lvl_wv]
        # index where level ii_lvl_wv begins
        n_start_wv_lvl = norm_indices[ii_lvl_wv] * n_scal
        # index where level ii_lvl_wv ends
        n_stop_wv_lvl = norm_indices[ii_lvl_wv + 1] * n_scal
        # index where propagators begin
        n_start_propa_lvl = n_start_propa[ii_lvl_wv]
        # index where propagators end
        n_stop_propa_lvl = n_start_propa[ii_lvl_wv + 1]
        # size of ONE propagator
        n_propa = int((n_stop_propa_lvl - n_start_propa_lvl) / q_wv)
        # size of the scaling function
        n_scal_pr = int(n_propa / (2 ** wv_ll))

        # --- LOOP ON THE WAVELET PARAMETERS OF EACH LEVEL --- #
        # for ii_wv in range(n_start_wv_lvl, n_stop_wv_lvl):
        for ii_wv in range(n_start_wv_lvl, n_stop_wv_lvl):

            # Value of the wavelet coefficient
            wv_coef = wv_x[ii_wv]
            # TODO optimize this?
            if wv_coef == 0:
                continue
            # choose the right propagator for the coefficient at current level = modulo of q_wv
            ii_q = ii_wv % q_wv

            # index where THE propagator begins
            n_start_propa_lvl_q = n_start_propa_lvl + ii_q * n_propa
            # index where THE propagator ends
            n_stop_propa_lvl_q = n_start_propa_lvl_q + n_propa

            # --- LOOP ON THE LEVELS OF THE PROPAGATOR / OUTPUT VECTOR --- #
            # write at level ii_lvl_pr in w_x_dx
            for ii_lvl_pr in range(0, wv_ll + 1):
                # current dilation level of the propagator
                q_pro = list_q[ii_lvl_pr]
                # index where the propagator level begins
                n_start_propa_lvl_q_lvl = n_start_propa_lvl_q + norm_indices[ii_lvl_pr] * n_scal_pr
                # index where the propagator level ends
                n_stop_propa_lvl_q_lvl = n_start_propa_lvl_q + norm_indices[ii_lvl_pr + 1] * n_scal_pr

                # index where level ii_lvl_wv_dx begins
                n_start_wvdx_lvl = norm_indices[ii_lvl_pr] * n_scal
                # index where level ii_lvl_wv_dx ends
                n_stop_wvdx_lvl = norm_indices[ii_lvl_pr + 1] * n_scal

                # number of coefficients at this level
                n_coef = n_stop_wvdx_lvl - n_start_wvdx_lvl

                # current ii_z takes into account dilations due to the change of levels.
                # must be calculated with respect to n_start_wv_lvl
                ii_wv_dilate = int(((ii_wv - ii_q) - n_start_wv_lvl) / q_wv * q_pro + n_start_wvdx_lvl)

                # size (and center) of the propagator
                n_ker = n_stop_propa_lvl_q_lvl - n_start_propa_lvl_q_lvl
                n_ker2 = n_ker // 2

                # we propagate the level ii_lvl_pr
                # the coef is included in the propagator
                # scan wv_x_dx in [ii_wv_dilate - n_ker/2, ii_wv_dilate + n_ker/2]. Remove points out of [n_start_wvdx_lvl, n_stop_wvdx_lvl-1]
                ind_min = max(ii_wv_dilate - n_ker2, n_start_wvdx_lvl)
                ind_max = min(ii_wv_dilate - n_ker2 + n_ker, n_stop_wvdx_lvl)

                # todo: search for nonzero coefficients
                for ii_ind in range(ind_min, ind_max):
                    # n_start_propa_lvl_q_lvl corresponds to the beginning of the propagator kernel
                    ii_ker = ii_ind - (ii_wv_dilate - n_ker2) + n_start_propa_lvl_q_lvl
                    propa_coef = wv_coef * dictionary[ii_ker]
                    # if abs(propa_coef) < v_p:
                    #     continue
                    wv_x_dx_cy[ii_ind] += propa_coef

    return wv_x_dx_cy



