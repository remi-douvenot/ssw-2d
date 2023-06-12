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
# @file ssw_2d_one_step.py
#
# @package: ssw_2d_one_step
# @author: R. Douvenot
# @date: 20/07/2021 last modif 20/07/2021
# @version: work in progress
#
# @brief One step of the ssw 2d free-space propagation
# @param[in] u_x Reduced electric field before the free-space propagation
# @param[in] library Pre-generated wavelet propagators.
# @param[in] config Structure containing the parameters of the simulation.
# @param[out] u_x_dx Reduced electric field after the free-space propagation
# @details Propagates a field with the 2D SSW technique in free space on one step. Decomposes the field, calls function
# "wavelet_propag_one_step" then recomposes the field from the wavelets.
##

# import cProfile, pstats, io
# def profile(fnc):
#     def inner(*args, **kwargs):
#         pr = cProfile.Profile()
#         pr.enable()
#         retval = fnc(*args, **kwargs)
#         pr.disable()
#         s = io.StringIO()
#         sortby = 'cumulative'
#         ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
#         ps.print_stats()
#         print(s.getvalue())
#         return retval
#
#     return inner
#
#
# @profile


import numpy as np
import pywt
import time
from src.wavelets.wavelet_operations import thresholding, q_max_calculation
from scipy.signal import convolve
from src.propa_cython.wavelet_propag_one_step import wavelet_propag_one_step_cy
from src.wavelets_cython.wavelets_operations import normalized_indices, calculate_dilation
import timeit


def ssw_2d_one_step(u_x, library, n_propa_lib, config):

    # Simulation parameters
    n_z = u_x.shape[0]

    # Wavelet parameters
    family = config.wv_family
    mode = 'per'

    # Decompose the field into a wavelet coefficient vector
    wv_x = pywt.wavedec(u_x, family, mode, config.wv_L)

    # Threshold V_s on the signal
    wv_x = thresholding(wv_x, config.V_s)

    # Propagation with the Python code
    if config.py_or_cy == 'Python':

        # Propagate the field in the wavelet domain
        wv_x_dx = wavelet_propag_one_step(wv_x, library, config)

    # Propagation with the Cython code
    elif config.py_or_cy == 'Cython':
        # # TODO: calculate library and nonzero coefficients at the beginning of the code
        # library2 = []
        # q_list = q_max_calculation(config.wv_L)
        # # Index at which the propagators of the level begin
        # n_propa_lib = np.zeros(config.wv_L + 2, dtype='int32')
        # # put the library in the shape of a vector. Useful for Cython version
        # for ii_lvl in range(0, config.wv_L + 1):
        #     n_propa_lib[ii_lvl + 1] += n_propa_lib[ii_lvl]  # add previous level
        #     for ii_q in range(0, q_list[ii_lvl]):
        #         toto = library[ii_lvl][ii_q]
        #         tata = pywt.coeffs_to_array(toto)[0]
        #         n_propa_lib[ii_lvl+1] += len(tata)  # add the size of each propagator
        #         library2 = np.append(library2, tata)
        wv_x, wx_x_shape = pywt.coeffs_to_array(wv_x)
        wv_x_dx_array = wavelet_propag_one_step_cy(n_z, wv_x, library, config.wv_L, n_propa_lib, config.V_p)

        # wv_x_dx2 = unsparsify_dok(wv_x_dx2_sparse)
        wv_x_dx = pywt.array_to_coeffs(wv_x_dx_array, wx_x_shape, output_format='wavedec')
    else:
        raise ValueError('py_or_cy variable can be ''Cython'' or ''Python'' only')

    # Threshold V_s on the signal
    wv_x_dx = thresholding(wv_x_dx, config.V_s)

    # Recompose signal
    u_x_dx = pywt.waverec(wv_x_dx, family, mode)

    return u_x_dx, wv_x_dx


##
# @package: wavelet_propag_one_step
# @author: Remi Douvenot
# @date: 20/07/2021 last modif 20/07/2021
# @version: V1.0
#
# @brief One step of the SSW 2D free-space propagation
# @param[in] wv_x Wavelet decomposition of the field before the free-space propagation
# @param[in] library Pre-generated wavelet propagators.
# @param[in] config Structure containing the parameters of the simulation.
# @param[out] wv_x_dx Wavelet decomposition of the field after the free-space propagation
# @details Apply the free-space propagators to all the nonzero wavelet coefficients
##
# import scipy


def wavelet_propag_one_step(wv_x, library, config):

    # Wavelet parameters
    family = config.wv_family
    mode = 'per'
    ll = config.wv_L

    n_z = config.N_z + config.N_im

    # --- Init the propagated wavelet coefficient --- #
    # Decompose a matrix of zeros to obtain a wavelet list full of zeros
    zeros = np.zeros(n_z, dtype='complex')  # matrix full of zeros
    # Wavelet transform
    wv_x_dx = pywt.wavedec(zeros, family, mode, ll)
    # changes the wavelet coeffs in complex 64 format.
    # also changes the tuple of the wavelet decomposition into lists that can be modified
    # wv_x_dx = fortran_type(wv_x_dx)
    # calculate the number of propagators q at each level
    list_q = q_max_calculation(ll)

    # ----------------------------------------------- #
    # ------ Propagation in the wavelet domain ------ #
    # --- Loop on all the (non-zero) coefficients --- #
    # ----------------------------------------------- #

    # @todo: multiprocessing
    # on level ii_lvl of the wavelet decomposition of the signal
    for ii_lvl in np.arange(0, ll+1):
        # print('Propagate wavelet level ', ii_lvl)

        # Compute q_max for ii_lvl
        q_max = list_q[ii_lvl]

        # wavelet positions of the coefficients to propagate (and coefficients values)
        wv_x_lvl = wv_x[ii_lvl]

        # loop on the propagators at level ii_lvl
        for ii_z in np.arange(0, q_max):
            # choose the propagator
            propagator = library[ii_lvl][ii_z]
            # extract the wavelets that match this propagator
            wv_x_lvl_z = wv_x_lvl[ii_z::q_max]
            wv_x_dx = add_propagator_at_once(wv_x_lvl_z, propagator, ll, wv_x_dx)
            # print('fill all orientation time for one coeff',t_end_or-t_start_or)
            # t_end_orp = time.process_time()
            # print('propagate all coeffs for one level one orientation time',t_end_orp-t_start_orp)
        # t_end_level = time.process_time()
        # print('propagate all coeffs for one level time',t_end_level-t_start_level)

    # --------------------- END --------------------- #
    # ------ Propagation in the wavelet domain ------ #
    # --- Loop on all the (non-zero) coefficients --- #
    # ----------------------------------------------- #

    return wv_x_dx

##
# @package: add_propagator_at_once
# @author: Remi Douvenot
# @date: 20/07/2021
# @version: V1.0
#
# @brief Add to wv_x_dx all the propagated wavelets corresponding to one propagator
# @param[in] wv_x_lvl Wavelet parameters that correspond to the propagator (chosen level and propagator number)
# @param[in] propagator Pre-generated wavelet propagator
# @param[in] ll Max level of the multiscale decomposition
# @param[in] wv_x_dx Wavelet coefficients calculated for wv_x_dx so far
# @param[out] wv_x_dx Wavelet coefficients calculated for wv_x_dx with the inputted "propagator" taken into account
# @details Add to wv_x_dx all the propagated wavelets corresponding to one propagator. For scaling function, the
# propagator is chosen wrt. level and translation. Then convolved by the approx or detail parameters of wv_x_dx
##


def add_propagator_at_once(wv_x_lvl, propagator, ll, wv_x_dx):

    # calculation of the dilation levels
    list_q = q_max_calculation(ll)
    # we add to U_d_dx the propagator affected with the coefficient
    # loop on the levels of the propagator
    for ii_lvl in range(0, ll + 1):
        # print('ii_lvl2 = ', ii_lvl)

        # Dilation level (the same for all orientations)
        t_level_prime = list_q[ii_lvl]
        # Size of the vector after dilation
        n_z_dilate = t_level_prime * wv_x_lvl.size

        # loop on wavelet orientations: one convolution of each orientation

        # initialisation of the input signal for convolution
        wv_x_lvl_dilate = np.zeros(n_z_dilate, dtype='complex')

        # first wavelet of each level corresponds to the input of the propagator
        wv_x_lvl_dilate[::t_level_prime] = wv_x_lvl

        # --- Python Convolution in Python --- #
        # @note This shift is necessary to "center" the convolution "convolve"
        propagator_lvl_or = np.zeros_like(propagator[ii_lvl])
        propagator_lvl_or[0:-1:] = propagator[ii_lvl][1::]
        # the output is the convolution of each nonzero parameter with the propagator
        wv_x_dx[ii_lvl] += convolve(wv_x_lvl_dilate, propagator_lvl_or, mode='same')

        # print('convolution is ',t_end-t_start, 's')

    return wv_x_dx


##
# @package: eliminate_top_field
# @author: R. Douvenot
# @date: 07/09/2021
# @version: V1.0
#
# @brief Eliminate the top field in the apodisation layer due to perdiodic decomposition
# @param[in] u_x Field
# @param[out] u_x Field with top wavelets = 0
##


# @ todo Code it in Fortran
def eliminate_top_field(u_x):

    # find the last zero
    zeros_indices = np.where(u_x == 0)[0]  # [0] because where gives a 1-dimensional tuple
    # print(len(zeros_indices))
    if len(zeros_indices) == 0:
        # @todo Print a warning!
        ii_zero = u_x.size
    else:
        # print(zeros_indices)
        # fill zeros up to this last value
        ii_zero = np.max(zeros_indices)
    u_x[ii_zero:-1] = 0

    return u_x


##
# @package: eliminate_top_wavelets
# @author: R. Douvenot
# @date: 09/06/2021
# @version: V1.0
#
# @brief Eliminate the top wavelets in the apodisation layer due to perdiodic decomposition
# @param[in] wv_x Wavelet decomposition
# @param[out] wv_x Sparse wavelet decomposition with top wavelets = 0
##

# @ todo Code it in Fortran
def eliminate_top_wavelets(wv_x):

    # max level of decomposition
    ll = len(wv_x)-1
    # on each level
    for ii_lvl in np.arange(0, ll+1):
        # array of wavelet coefs
        wv_x_lvl = wv_x[ii_lvl]
        # find the last zero
        ii_zero = np.max(np.where(wv_x_lvl == 0))
        # fill zeros up to this last value
        wv_x[ii_lvl][ii_zero:-1] = 0

    return wv_x

##
# @package: fortran_type
# @author: R. Douvenot
# @date: 09/06/2021
# @version: V1.0
#
# @brief Put a wavelet decomposition in a Fortran format
# @param[in] wv_x Wavelet decomposition
# @param[out] wv_x_fortran Wavelet decomposition
##


def fortran_type(wv_u):

    # max level of decomposition
    ll = len(wv_u)-1
    # creation of the empty list
    wv_u_fortran = [[]] * (ll+1)
    # fill the wavelet levels
    for ii_lvl in np.arange(0, ll+1):
        wv_u_fortran[ii_lvl] = wv_u[ii_lvl]

    return wv_u_fortran


def wavelet_propag_one_step3(wv_x, library, wv_ll, n_start_propa):

    # --- Init the propagated wavelet coefficient --- #
    # OUTPUT: matrix full of zeros: wavelet decomposition after propagation
    n_z = len(wv_x)
    wv_x_dx = np.zeros(n_z, dtype=np.complex128)
    # INPUTS (constant)
    # define variables
    n_scal = int(len(wv_x) / (2**wv_ll))  # number of parameters on the highest levels (scaling function) of the signal vectors

    # lists of the number of propagators per level
    list_q = calculate_dilation(wv_ll)
    # list_indices = normalized_indices(wv_ll)

    # ----------------------------------------------------------- #
    # ------------- Propagation in the wavelet domain ----------- #
    # --- LOOP ON THE WAVELET LEVELS OF THE INPUT VECTOR WV_X --- #
    # ----------------------------------------------------------- #

    # --- LOOP ON THE LEVELS OF W_X --- #
    for ii_lvl_wv in range(0, wv_ll + 1):
        # number of propagators at this level
        q_wv = list_q[ii_lvl_wv]
        norm_indices = normalized_indices(wv_ll)
        # index where level ii_lvl_wv begins
        n_start_wv_lvl = norm_indices[ii_lvl_wv] * n_scal
        # index where level ii_lvl_wv ends
        n_stop_wv_lvl = norm_indices[ii_lvl_wv+1] * n_scal
        # TODO: keep only nonzero coefs
        # nz_indices = wv_x[ii_lvl_wv].nonzero()[1]
        # index where propagators begin
        n_start_propa_lvl = n_start_propa[ii_lvl_wv]
        # index where propagators end
        n_stop_propa_lvl = n_start_propa[ii_lvl_wv+1]
        # size of ONE propagator
        n_propa = int((n_stop_propa_lvl-n_start_propa_lvl) / q_wv)
        # size of the scaling function
        n_scal_pr = int(n_propa / (2**wv_ll))
        # --- LOOP ON THE WAVELET PARAMETERS OF EACH LEVEL --- #
        for ii_wv in range(n_start_wv_lvl, n_stop_wv_lvl):
            # Value of the wavelet coefficient
            wv_coef = wv_x[ii_wv]
            if wv_coef == 0:
                continue
            # choose the right propagator for the coefficient at current level = modulo of q_wv
            ii_q = ii_wv % q_wv

            # index where THE propagator begins
            n_start_propa_lvl_q = n_start_propa_lvl + ii_q*n_propa
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
                n_stop_propa_lvl_q_lvl = n_start_propa_lvl_q + norm_indices[ii_lvl_pr+1] * n_scal_pr

                # index where level ii_lvl_wv_dx begins
                n_start_wvdx_lvl = norm_indices[ii_lvl_pr] * n_scal
                # index where level ii_lvl_wv_dx ends
                n_stop_wvdx_lvl = norm_indices[ii_lvl_pr+1] * n_scal

                # number of coefficients at this level
                n_coef = n_stop_wvdx_lvl-n_start_wvdx_lvl

                # current ii_z takes into account dilations due to the change of levels.
                # must be calculated with respect to n_start_wv_lvl
                ii_z_current = int(((ii_wv - ii_q)-n_start_wv_lvl) / q_wv * q_pro + n_start_wvdx_lvl)

                # size (and center) of the propagator
                n_ker = n_stop_propa_lvl_q_lvl - n_start_propa_lvl_q_lvl
                n_ker2 = n_ker//2

                # we propagate the level ii_lvl_pr
                # the coef is included in the propagator
                uu = np.array(library[n_start_propa_lvl_q_lvl:n_stop_propa_lvl_q_lvl])
                propagator_lvl = wv_coef * uu

                # scan wv_x_dx in [ii_z_current - n_ker/2, ii_z_current + n_ker/2]. Remove points out of [0,n_coef]
                ind_min = int(max(n_start_wvdx_lvl, ii_z_current - n_ker2))
                ind_max = int(min(n_start_wvdx_lvl+n_coef, ii_z_current - n_ker2 + n_ker))

                # print("ind_min = ", ind_min)
                # print("ind_max = ", ind_max)
                # loop on the propagators coefficients
                for ii_ind in range(ind_min, ind_max):
                    wv_x_dx[ii_ind] += propagator_lvl[ii_ind-ind_min]

    return wv_x_dx
