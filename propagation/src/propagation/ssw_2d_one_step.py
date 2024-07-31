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
# from src.propa_cython.wavelet_propag_one_step import wavelet_propag_one_step_cy # imported dynamically
# from src.wavelets_cython.wavelets_operations import normalized_indices, calculate_dilation
import timeit
import importlib


def ssw_2d_one_step(u_x, dictionary, n_propa_lib, config):

    # Simulation parameters
    n_z = u_x.shape[0]

    # Wavelet parameters
    family = config.wv_family
    mode = 'per'

    # Decompose the field into a wavelet coefficient vector
    wv_x = pywt.wavedec(u_x, family, mode, config.wv_L)

    # # Threshold V_s on the signal
    # wv_x = thresholding(wv_x, config.V_s)

    # Propagation with the Python code
    if config.py_or_cy == 'Python':
        # Threshold V_s on the signal
        wv_x = thresholding(wv_x, config.V_s)
        # Propagate the field in the wavelet domain
        wv_x_dx = wavelet_propag_one_step(wv_x, dictionary, config)
        # Threshold V_s on the signal
        wv_x_dx = thresholding(wv_x_dx, config.V_s)

    # Propagation with the Cython code
    elif config.py_or_cy == 'Cython':
        # Compile if necessary and import cython function
        one_step_cy = importlib.import_module('src.propa_cython.wavelet_propag_one_step')
        # put coefficients in array shape
        wv_x, wx_x_shape = pywt.coeffs_to_array(wv_x)
        # Apply the threshold on the array
        wv_x = pywt.threshold(wv_x, config.V_s, mode='hard')
        # Propagate the field in the wavelet domain
        wv_x_dx_array = one_step_cy.wavelet_propag_one_step_cy(n_z, wv_x, dictionary, config.wv_L, n_propa_lib, config.V_p)
        # Apply the threshold on the array
        wv_x_dx_array = pywt.threshold(wv_x_dx_array, config.V_s, mode='hard')
        # back in coefficients shape
        wv_x_dx = pywt.array_to_coeffs(wv_x_dx_array, wx_x_shape, output_format='wavedec')
    else:
        raise ValueError('py_or_cy variable can be ''Cython'' or ''Python'' only')

    # # Threshold V_s on the signal
    # wv_x_dx = thresholding(wv_x_dx, config.V_s)

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
# @param[in] dictionary Pre-generated wavelet propagators.
# @param[in] config Structure containing the parameters of the simulation.
# @param[out] wv_x_dx Wavelet decomposition of the field after the free-space propagation
# @details Apply the free-space propagators to all the nonzero wavelet coefficients
##
# import scipy


def wavelet_propag_one_step(wv_x, dictionary, config):

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
            propagator = dictionary[ii_lvl][ii_z]
            # extract the wavelets that match this propagator
            wv_x_lvl_z = wv_x_lvl[ii_z::q_max]
            wv_x_dx = add_propagator_at_once(wv_x_lvl_z, propagator, ll, wv_x_dx)
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
