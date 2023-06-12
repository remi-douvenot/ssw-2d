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
# @file wwp_2d_one_step.py
#
# @package: wwp_2d_one_step
# @author: R. Douvenot
# @date: 08/11/2022
# @version: work in progress
#
# @brief One step of the wwp 2d free-space propagation
# @param[in] w_x Wavelet decomposition of the electric field before the free-space propagation
# @param[in] dictionary Pre-generated wavelet propagators.
# @param[in] config Structure containing the parameters of the simulation.
# @param[out] w_x_dx Wavelet decomposition of the electric field after the free-space propagation
# @details Propagates a field decomposed on a wavelet based with the 2D WWP technique in free space on one step.
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
import multiprocessing as mp
import pywt
import time
from src.wavelets.wavelet_operations import thresholding, q_max_calculation
from src.propagation.ssw_2d_one_step import add_propagator_at_once
from scipy.signal import convolve


# todo: Cython for WWP
def wwp_2d_one_step(wv_x, dictionary, config):

    # Propagate the field in the wavelet domain
    wv_x_dx = wavelet_propag_one_step(wv_x, dictionary, config)

    # Threshold V_s on the signal
    wv_x_dx = thresholding(wv_x_dx, config.V_s)

    return wv_x_dx


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
            wv_x_dx = add_propagator_at_once(wv_x_lvl_z, propagator, config.wv_L, wv_x_dx)
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
