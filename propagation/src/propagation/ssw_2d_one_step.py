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
import multiprocessing as mp
import pywt
import time
from src.wavelets.thresholding import thresholding
from src.propagators.library_generation import q_max_calculation
from scipy.sparse import coo_matrix  # for sparsify
# from src.maths import convolution  # for add_propagator_at_once
from scipy.signal import convolve


def ssw_2d_one_step(u_x, library, config):
    # Simulation parameters
    n_z = u_x.shape

    # Wavelet parameters
    family = config.wv_family
    mode = 'per'

    # Decompose the field into a wavelet coefficient vector
    wv_x = pywt.wavedec(u_x, family, mode, config.wv_L)

    # Threshold V_s on the signal
    wv_x = thresholding(wv_x, config.V_s)

    # Propagate the field in the wavelet domain
    wv_x_dx = wavelet_propag_one_step(wv_x, library, config)

    # Threshold V_s on the signal
    wv_x_dx = thresholding(wv_x_dx, config.V_s)

    # @todo Eliminate spurious field on top of the apodisation layer only if there is a ground
    # These spurious wavelets are due to bottom field decomposed with periodic mode
    # if config.ground == 'PEC' or config.ground == 'dielectric':
    #     wv_x_dx = eliminate_top_wavelets(wv_x_dx)
    # Calculate the field from the wavelet decomposition
    # @todo Eliminate spurious field on top of the apodisation layer only if there is a ground
    # This spurious field is due to bottom wavelets recomposed with periodic mode
    u_x_dx = pywt.waverec(wv_x_dx, family, mode)
    # if config.ground == 'PEC' or config.ground == 'dielectric':
    #     u_x_dx = eliminate_top_field(u_x_dx)

    # store a sparse vector
    wv_x_dx_sparse = sparsify(wv_x_dx)

    return u_x_dx, wv_x_dx_sparse


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
    zeros = np.zeros(n_z, dtype='complex64')  # matrix full of zeros
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
        wv_x_lvl_dilate = np.zeros(n_z_dilate, dtype='complex64')

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
# @package: sparsify
# @author: R. Douvenot
# @date: 09/06/2021
# @version: V1.0
#
# @brief Put a wavelet decomposition in a sparse shape (coo format)
# @param[in] wv_x Wavelet decomposition
# @param[out] wv_x_sparse Sparse wavelet decomposition
##


def sparsify(wv_x):

    # max level of decomposition
    ll = len(wv_x)-1
    # creation of the empty list
    wv_x_sparse = [[]] * (ll+1)
    # fill the scaling function
    # fill the wavelet levels
    for ii_lvl in np.arange(0, ll+1):
        wv_x_sparse[ii_lvl] = coo_matrix(wv_x[ii_lvl])

    return wv_x_sparse

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
# @brief Put a wavelet decomposition in a Fortran format (with complex64 instead of complex128)
# @param[in] wv_x Wavelet decomposition
# @param[out] wv_x_fortran Wavelet decomposition with complex64 format
##


def fortran_type(wv_u):

    # max level of decomposition
    ll = len(wv_u)-1
    # creation of the empty list
    wv_u_fortran = [[]] * (ll+1)
    # fill the wavelet levels
    for ii_lvl in np.arange(0, ll+1):
        wv_u_fortran[ii_lvl] = wv_u[ii_lvl].astype('complex64')

    return wv_u_fortran
