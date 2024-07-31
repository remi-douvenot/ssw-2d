# This file is part of SSW-2D.
# SSW-2D is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# SSW-2D is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
# of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Foobar. If not, see
# <https://www.gnu.org/licenses/>.
import matplotlib.pyplot as plt
##
# @file wwp_2d_cy.py
#
# @author R. Douvenot
# @package WWP_2D_CY
# @date 04/07/23
# @version In progress
# @brief The core of the 2D WWP in Cython: propagates a 1D initial field with WWP in 2D.
# @details This function is the core of the 2D WWP code. It propagates a 1D initial reduced field with WWP in 2D,
# in a Cartesian environment. The output field is given in the wavelet domain for a reduced memory size. \n
# The library of propagators is computed. Then the field is propagated in the wavelet domain using this library. \n
# The refractivity is taken into account with a phase screen applied on wavelets directly.
# Same for apodisation.
#
# More details in:
# Hang Zhou, Alexandre Chabory, Rémi Douvenot. A Fast Wavelet-to-Wavelet Propagation Method for the Simulation of
# Long-Range Propagation in Low Troposphere. IEEE Transactions on Antennas and Propagation, 2022, 70, pp.2137-2148
#
# @param[in] u_0 Initial field. Complex array of size (N_z)
# @param[in] config Class that contains the configuration. See class list for details
# @param[in] config_source Class that contains the source configuration. See class list for details
# @param[in] n_refraction Vector containing the modified refractive index n. Real array of size (N_z)
# @param[in] z_relief Vector containing the relief indices at each distance step. Real array of size (N_z)
# @param[out] wv_total Wavelet coefficients of the 3D field. Complex array of size (N_x, N_z)
# @warning Put lengths = multiple of 2^l
##



import numpy as np
import time
import scipy.constants as cst
from src.propagators.dictionary_generation import dictionary_generation
from src.propagation.apodisation import apply_apodisation, apodisation_window
import pywt
# from src.wavelets.wavelet_operations import thresholding
from src.wavelets.wavelet_operations import sparsify  # for sparsify
from src.wavelets_cython.wavelets_operations import calculate_dilation
from src.propa_cython.wavelet_propag_one_step import wavelet_propag_one_step_cy
from cython cimport boundscheck, wraparound, cdivision, nonecheck

@boundscheck(False)
@wraparound(False)
@cdivision(True)
@nonecheck(False)

def wwp_2d_cy(const double complex[:] u_0, config, const double[:] n_refraction):


    # Simulation parameters
    cdef int n_x = config.N_x
    cdef int n_z = config.N_z
    cdef int wv_ll = config.wv_L
    cdef double v_p = config.V_p
    cdef double complex[:] dictionary_cy
    cdef double k0 = 2*np.pi*config.freq/cst.c
    cdef double complex[:] phase_screen_wavelets = np.zeros(n_z, dtype=np.complex128)
    # indices in loops
    cdef Py_ssize_t ii_lvl
    # vectors
    cdef double complex[:] propagator_vect
    cdef double complex[:] wv_x
    cdef double complex[:] wv_x_dx
    cdef int[:] shape_wv_x = np.zeros(wv_ll+1, dtype=np.int32)

    print("frequence = ", config.freq)
    # Compute the dictionary of unit operators. Those operators correspond to the wavelet-to-wavelet propagation of
    # each basis wavelet.
    print('--- Creation of the dictionary of the free-space operators ---')
    t_dictionary_s = time.process_time()
    dictionary2 = dictionary_generation(config)
    t_dictionary_f = time.process_time()
    print('--- END Creation of the dictionary of the free-space operators ---')
    print('--- Dedicated time (s) ---', np.round(t_dictionary_f - t_dictionary_s, decimals=3))
    print(' ')
    # save the final electric field
    np.save('./outputs/dictionary', dictionary2)
    # in Cython, the dictionary is stored in the shape of one unique vector.
    dictionary = []
    cdef int[:] q_list = calculate_dilation(wv_ll)
    # Index at which the propagators of the level begin
    cdef int[:] n_propa_lib = np.zeros(wv_ll + 2, dtype='int32')
    # put the library in the shape of a vector. Useful for Cython version
    for ii_lvl in range(0, wv_ll + 1):
        n_propa_lib[ii_lvl + 1] += n_propa_lib[ii_lvl]  # add previous level
        for ii_q in range(0, q_list[ii_lvl]):
            propagator_list = dictionary2[ii_lvl][ii_q]
            propagator_vect = pywt.coeffs_to_array(propagator_list)[0]
            n_propa_lib[ii_lvl + 1] += len(propagator_vect)  # add the size of each propagator
            dictionary = np.append(dictionary, propagator_vect)
    nonzero_ind = np.nonzero(dictionary)

    dictionary_cy = dictionary

    # --- Sizes of the apodisation and image layers --- #
    if config.ground == 'PEC' or config.ground == 'Dielectric':
        n_im = np.int32(np.round(config.N_z * config.image_layer))
        remain_im = n_im % 2**wv_ll
        if remain_im != 0:
            n_im += 2**wv_ll - remain_im
    else:  # config.ground == 'None':
        print('--- Main loop. No ground ---')
        n_im = 0
    config.N_im = n_im

    n_apo_z = np.int32(config.apo_z * config.N_z)
    remain_apo = n_apo_z % 2 ** wv_ll
    if remain_apo != 0:
        n_apo_z += 2 ** wv_ll - remain_apo
    # ------------------------------------------------- #

    # --- Creation of the first field and apodisation window --- # @todo Code other apodisation windows
    # along z
    apo_window_z = apodisation_window(config.apo_window, n_apo_z)
    # initial field
    u_0 = apply_apodisation(u_0, apo_window_z, config)
    # ------------------------------------------ #

    # --- Put initial field in the wavelet domain --- #

    # Decompose the initial field into a wavelet coefficient vector
    wv_0 = pywt.wavedec(np.array(u_0), config.wv_family, 'per', wv_ll)
    wv_x, shape_wv_x_pywt = pywt.coeffs_to_array(wv_0)
    # stop indices of each level
    shape_wv_x[0] = shape_wv_x_pywt[0][0].stop
    for ii_ll in range(1, wv_ll+1):
        shape_wv_x[ii_ll] = shape_wv_x_pywt[ii_ll]['d'][0].stop
    # Threshold V_s on the signal
    wv_x = pywt.threshold(wv_x, v_p, mode='hard')

    # --- Put apodisation window in the wavelet domain --- #
    apo_window_wavelets = reshape_apodisation_on_wavelets(apo_window_z, config.ground, shape_wv_x)
    # --- Put refraction window in the wavelet domain --- #
    n_refraction_wavelets = reshape_refraction_on_wavelets(n_refraction, shape_wv_x)
    # --- From refraction to phase screen --- #
    for ii_z in range(0,n_z):
        phase_screen_wavelets[ii_z] = np.exp(-1j * k0 * (n_refraction_wavelets[ii_z]-1) * config.x_step)

    # saved total wavelet parameters (sparse coo matrix)
    wv_total = [[]] * n_x
    # ----------------------- #

    # Loop over the x_axis
    for ii_x in np.arange(1, n_x+1):
        if ii_x % 100 == 0:
            print('Iteration', ii_x, '/', n_x, '. Distance =', ii_x*config.x_step)
        # --- apodisation applied on wavelets --- #
        for ii_z in range(0,n_z):
            wv_x[ii_z] = wv_x[ii_z] * apo_window_wavelets[ii_z]
        # --------------------------------------- #

        # ------------------------------ #
        # --- Free-space propagation --- #
        # ------------------------------ #
        if config.ground == 'Dielectric' or config.ground == 'PEC':

            raise ValueError(['Dielectric and PEC ground not available with WWP'])

        elif config.ground == 'None':

            # Propagate the field in the wavelet domain
            wv_x_dx = wavelet_propag_one_step_cy(n_z, wv_x, dictionary_cy, wv_ll, n_propa_lib, v_p)

            # Threshold V_s on the signal
            wv_x_dx = pywt.threshold(wv_x_dx, v_p, mode='hard')

        else:
            raise ValueError(['Ground condition should be dielectric, PEC, or None'])
        # ---------- END --------------- #
        # --- Free-space propagation --- #
        # ------------------------------ #

        # --- refraction applied on wavelets --- #
        for ii_z in range(0, n_z):
            wv_x_dx[ii_z] = wv_x_dx[ii_z] * phase_screen_wavelets[ii_z]
        # --------------------------------------- #

        # update w_x
        for ii_z in range(0, n_z):
            wv_x[ii_z] = wv_x_dx[ii_z]
        # store the wavelet parameters (in coo format)
        wv_store = pywt.array_to_coeffs(wv_x_dx, shape_wv_x_pywt, output_format='wavedec')
        wv_total[ii_x-1] = sparsify(wv_store)

    # back from wavelet to field
    u_last = pywt.waverec(wv_store, config.wv_family, mode='per')

    return u_last, wv_total

##
# @brief function that projects the apodization function on the wavelet basis
# @author R. Douvenot
# @package reshape_apodisation_on_wavelets
# @date 04/07/23
# @version OK
#
# @details Function that projects the apodization function on the wavelet basis when represented as a single array.
#
# def reshape_apodisation_on_wavelets(apo_window_z, ground, shape_wv_x):
#
# @params[in] apo_window_z : apodisation window wrt z
# @params[in] ground : string of characters. No ground, PEC or Dielectric
# @params[in] shape_wv_x : shape of the wavelet vector (position of the wavelet levels in the vector)
# @params[out] apo_wavelets : apodisation window projected on the wavelet levels in the same shape on the wavelets vector
##

@boundscheck(False)
@wraparound(False)
@cdivision(True)
@nonecheck(False)

cdef reshape_apodisation_on_wavelets(const double[:] apo_window_z, ground, const int[:] shape_wv_x):

    # wavelet level
    cdef int wv_ll = shape_wv_x.shape[0]-1
    cdef int shape_prev = 0
    # number of q_max per level
    cdef int[:] q_max = calculate_dilation(wv_ll)
    cdef int[:] decimation = np.zeros(wv_ll+2, dtype=np.int32)
    cdef int delta
    # decimation coefficient per level
    for ii in range(0, wv_ll+1):
        decimation[ii] = int(2 ** wv_ll / q_max[ii])
    # total size of the vector
    cdef int n_z = shape_wv_x[wv_ll]
    # size of the apodisation function
    cdef int n_apo_z = apo_window_z.size
    cdef double[:] apo_wavelets = np.ones(n_z, dtype=np.float64)
    # size of the current level
    cdef int n_lvl
    # indices in loops
    cdef Py_ssize_t ii_lvl, ii_wz, ii_apo

    # on each level, apodize the edge(s)
    for ii_lvl in range(0,wv_ll+1):
        n_lvl = shape_wv_x[ii_lvl]
        # decimation level at the current level
        delta = decimation[ii_lvl]
        # number of coefficients on which applying apodisation
        n_apo_z_delta = int(n_apo_z / delta)
        #
        for ii_wz in range(0, n_apo_z_delta):
            # on the last points of the level
            ii_apo = n_lvl-n_apo_z_delta + ii_wz
            # apply apodization with proper subsampling
            apo_wavelets[ii_apo] = apo_window_z[delta*ii_wz]

        if ground == 'None':
            for ii_wz in range(0, n_apo_z_delta):
                # on the first points of the level
                ii_apo = n_apo_z_delta - ii_wz + shape_prev - 1
                # apply apodization with proper subsampling
                apo_wavelets[ii_apo] = apo_window_z[delta * ii_wz]
            # update the previous vector length
            shape_prev = shape_wv_x[ii_lvl]

    return apo_wavelets


##
# @brief function that projects the refraction vertical profile on the wavelet basis
# @author R. Douvenot
# @package reshape_refraction_on_wavelets
# @date 04/07/23
# @version OK
#
# @details Function that projects the apodization function on the wavelet basis when represented as a single array.
#
# def reshape_refraction_on_wavelets(apo_window_z, ground, shape_wv_x):
#
# @params[in] n_profile : refraction profile wrt z
# @params[in] shape_wv_x : shape of the wavelet vector (position of the wavelet levels in the vector)
# @params[out] apo_wavelets : apodisation window projected on the wavelet levels in the same shape on the wavelets vector
##

@boundscheck(False)
@wraparound(False)
@cdivision(True)
@nonecheck(False)

cdef reshape_refraction_on_wavelets(const double[:] n_profile, const int[:] shape_wv_x):

    # wavelet level
    cdef int wv_ll = shape_wv_x.shape[0]-1
    # number of q_max per level
    cdef int[:] q_max = calculate_dilation(wv_ll)
    cdef int[:] decimation = np.zeros(wv_ll+2, dtype=np.int32)
    cdef int delta
    # decimation coefficient per level
    for ii in range(0, wv_ll+1):
        decimation[ii] = int(2 ** wv_ll / q_max[ii])
    # total size of the vector
    cdef int n_z = shape_wv_x[wv_ll]
    # size of the apodisation function
    cdef int n_ref_z = n_profile.size
    cdef double[:] ref_wavelets = np.ones(n_z, dtype=np.float64)
    # size of the current level
    cdef int n_lvl
    # indices in loops
    cdef Py_ssize_t ii_lvl, ii_wz, ii_ref

    # on each level, apodize the edge(s)
    for ii_lvl in range(0,wv_ll+1):
        n_lvl = shape_wv_x[ii_lvl]
        # decimation level at the current level
        delta = decimation[ii_lvl]
        # number of coefficients on which applying apodisation
        n_ref_z_delta = int(n_ref_z / delta)
        #
        for ii_wz in range(0, n_ref_z_delta):
            # on the last points of the level
            ii_ref = n_lvl-n_ref_z_delta + ii_wz
            # apply apodization with proper subsampling
            ref_wavelets[ii_ref] = n_profile[delta*ii_wz]

    return ref_wavelets
