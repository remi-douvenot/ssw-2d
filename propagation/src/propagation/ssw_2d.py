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
# @file ssw_2d.py
#
# @author R. Douvenot
# @package SSW_2D
# @date 06/02/19, modif 28/02/2022
# @version In progress
# @brief The core of the 2D SSW: propagates a 1D initial field with SSW in 2D.
# @details This function is the core of the 2D SSW code. It propagates a 1D initial reduced field with SSW in 2D,
# in a Cartesian environment. The output field is given in the wavelet domain for a reduced memory size. \n
# The library of propagators is computed. Then the field is propagated in the wavelet domain using this library. \n
# The refractivity is taken into account with a phase screen. Apodisation is also applied at each step.
#
# @param[in] u_0 Initial field. Complex array of size (N_z)
# @param[in] config Class that contains the configuration. See class list for details
# @param[in] config_source Class that contains the source configuration. See class list for details
# @param[in] n_refraction Vector containing the modified refractive index n. Real array of size (N_z)
# @param[in] z_relief Vector containing the relief indices at each distance step. Real array of size (N_z)
# @param[out] U_tot Wavelet coefficients of the 3D field. Complex array of size (N_x, N_z)
# @warning Put lengths = multiple of 2^l
##

# u_x_dx = SSW(u_0,simulation_parameters)
#
#######################################################################################################################

import numpy as np
import time
import scipy.constants as cst
import pywt
from src.propagators.dictionary_generation import dictionary_generation
from src.propagation.ssw_2d_one_step import ssw_2d_one_step
from src.propagation.apodisation import apply_apodisation, apodisation_window
from src.atmosphere.genere_n_profile import genere_phi_turbulent
from src.DSSF.dmft import dmft_parameters, u2w, w2u, surface_wave_propagation
from src.propagation.refraction import apply_refractive_index
from src.propagation.image_field import compute_image_field, compute_image_field_tm_pec
from src.wavelets.wavelet_operations import sparsify, q_max_calculation
# from src.propa_cython.refraction_cy import apply_refractive_index_cy # import not used


def ssw_2d(u_0, config, n_refraction, ii_vect_relief):

    # Simulation parameters
    n_x = config.N_x

    # --- Creation of the apodisation window --- # @todo Code other apodisation windows
    # along z
    n_apo_z = int(config.apo_z * config.N_z)
    apo_window_z = apodisation_window(config.apo_window, n_apo_z)
    # ------------------------------------------ #

    # Compute the dictionary of unit operators. Those operators correspond to the wavelet-to-wavelet propagation of
    # each basis wavelet.
    print('--- Creation of the dictionary of the free-space operators ---')
    t_dictionary_s = time.process_time()
    dictionary = dictionary_generation(config)
    t_dictionary_f = time.process_time()
    print('--- END Creation of the dictionary of the free-space operators ---')
    print('--- Dedicated time (s) ---', np.round(t_dictionary_f - t_dictionary_s, decimals=2))
    print(' ')
    # save the final electric field
    print(len(dictionary), len(dictionary[0]), len(dictionary[1]), len(dictionary[2]))
    # np.save('./outputs/dictionary', dictionary)
    n_propa_lib = 0  # not used in Python code
    # if Cython, the dictionary is stored in the shape of one unique vector.
    if config.py_or_cy == 'Cython':
        dictionary2 = []
        q_list = q_max_calculation(config.wv_L)
        # Index at which the propagators of the level begin
        n_propa_lib = np.zeros(config.wv_L + 2, dtype='int32')
        # put the library in the shape of a vector. Useful for Cython version
        for ii_lvl in range(0, config.wv_L + 1):
            n_propa_lib[ii_lvl + 1] += n_propa_lib[ii_lvl]  # add previous level
            for ii_q in range(0, q_list[ii_lvl]):
                propagator_list = dictionary[ii_lvl][ii_q]
                propagator_vect = pywt.coeffs_to_array(propagator_list)[0]
                n_propa_lib[ii_lvl + 1] += len(propagator_vect)  # add the size of each propagator
                dictionary2 = np.append(dictionary2, propagator_vect)
        dictionary = dictionary2
        # nonzero_ind = np.nonzero(dictionary)

    # --- Sizes of the apodisation and image layers --- #
    if config.ground == 'PEC' or config.ground == 'Dielectric':
        n_im = int(np.round(config.N_z * config.image_layer))
        remain_im = n_im % 2**config.wv_L
        if remain_im != 0:
            n_im += 2**config.wv_L - remain_im
    else:  # config.ground == 'None':
        print('--- Main loop. No ground ---')
        n_im = 0
    config.N_im = n_im

    # ------------------------------------------------- #

    # --- Initialisations --- #
    # initial field
    u_x = apply_apodisation(u_0, apo_window_z, config)
    if config.ground == 'Dielectric' or config.ground == 'PEC':
        # derivative of the relief calculated once for all
        diff_relief = np.diff(ii_vect_relief)
    # field after delta x
    u_x_dx = np.zeros_like(u_x)
    # total wavelet parameters (sparse coo matrix)
    wv_total = [[]] * n_x
    # ----------------------- #

    # Loop over the x_axis
    for ii_x in np.arange(1, n_x+1):
        if ii_x % 100 == 0:
            print('Iteration', ii_x, '/', n_x, '. Distance =', ii_x*config.x_step)
        # --- apodisation --- #
        u_x = apply_apodisation(u_x, apo_window_z, config)
        # ------------------- #

        # --- refractivity applied twice 1/2 --- #
        # if config.py_or_cy == 'Python':
        u_x = apply_refractive_index(u_x, n_refraction, config)
        # else:  # config.py_or_cy == 'Cython':
        #     u_x = np.array(apply_refractive_index_cy(u_x, n_refraction, config.freq, config.x_step))

        # -------------------------------------- #

        # ------------------------------ #
        # --- Free-space propagation --- #
        # ------------------------------ #
        if config.ground == 'Dielectric':

            # descending relief
            if diff_relief[ii_x - 1] < 0:
                # Add zeros below the field
                u_x = shift_relief(u_x, -diff_relief[ii_x - 1])

            # DMFT parameters
            alpha, r0, aa = dmft_parameters(ii_x, config)

            # from u to w
            w_x = u2w(alpha, u_x, config.N_z, config.z_step)

            # Add the image layer to w to apply reflection
            w_x = compute_image_field(w_x, n_im)

            # Propagate using SSW
            w_x_dx, wavelets_x_dx = ssw_2d_one_step(w_x, dictionary, n_propa_lib, config)

            # Pop image field: remove the image points
            w_x_dx = w_x_dx[n_im:n_im + config.N_z]

            # propagate surface wave
            spectrum_w_0, spectrum_w_n_z = surface_wave_propagation(w_x_dx, config, r0, aa)
            # spectrum_w_0, spectrum_w_n_z = 0.0, 0.0
            # from w to u
            u_x_dx = w2u(spectrum_w_0, spectrum_w_n_z, w_x_dx, config.N_z, config.z_step, r0, aa)

            # ascending relief
            if diff_relief[ii_x - 1] > 0:
                # Put zeros in the relief
                u_x_dx = shift_relief(u_x_dx, -diff_relief[ii_x - 1])
                # u_x_dx[0:diff_relief[ii_x - 1]] = 0.0

        elif config.ground == 'PEC':

            # descending relief
            if diff_relief[ii_x - 1] < 0:
                # Add zeros below the field
                u_x = shift_relief(u_x, -diff_relief[ii_x - 1])

            # Add the image layer to apply the local image method #
            if config.polar == 'TE':
                u_x = compute_image_field(u_x, n_im)
            elif config.polar == 'TM':
                u_x = compute_image_field_tm_pec(u_x, n_im)
            # Propagate using SSW
            u_x_dx, wavelets_x_dx = ssw_2d_one_step(u_x, dictionary, n_propa_lib, config)
            # Pop image field: remove the image points
            u_x_dx = u_x_dx[n_im:n_im + config.N_z]
            # ascending relief
            if diff_relief[ii_x - 1] > 0:
                # Put zeros in the relief
                u_x_dx = shift_relief(u_x_dx, -diff_relief[ii_x - 1])
                # u_x_dx[0:diff_relief[ii_x - 1]] = 0.0

                # end of the loop
        elif config.ground == 'None':
            # print('No ground')

            # Propagate using SSW
            u_x_dx, wavelets_x_dx = ssw_2d_one_step(u_x, dictionary, n_propa_lib, config)

        else:
            raise ValueError(['Ground condition should be dielectric, PEC, or None'])
        # ---------- END --------------- #
        # --- Free-space propagation --- #
        # ------------------------------ #

        # --- refractivity applied twice 2/2 --- #
        # if config.py_or_cy == 'Python':
        u_x_dx = apply_refractive_index(u_x_dx, n_refraction, config)
        # else:  # config.py_or_cy == 'Cython':
        #     u_x_dx = apply_refractive_index_cy(u_x_dx, n_refraction, config.freq, config.x_step)
        if config.turbulence == 'Y':
            phi_turbulent = genere_phi_turbulent(config)
            u_x_dx = apply_phi_turbulent(u_x_dx, phi_turbulent, config)

        # -------------------------------------- #

        # store the wavelet parameters (in coo format)
        wv_total[ii_x-1] = sparsify(wavelets_x_dx)
        # store the wavelet parameters (in coo format)
        # spectrum_w_0_tot[ii_x - 1] = spectrum_w_0_tot
        # update u_x
        u_x = u_x_dx

    return u_x_dx, wv_total


def apply_phi_turbulent(u_x, phi_turbulent,config):

    # apply the turbulent phase screen of one step delta_x
    # half the refraction applied before and after propagation
    k0 = 2 * cst.pi * config.freq / cst.c
    u_x *= np.exp(-1j * phi_turbulent)

    return u_x

##
# @brief function that shifts a field (upwards or downwards)
# @author R. Douvenot
# @package apply_refractive_index
# @date 10/09/21
# @version OK
#
# @details Function unction that shifts a field (upwards or downwards).
# Preallocate zeros array then fill with shifted vector
#
# @params[in] arr : input array (complex array)
# @params[in] num : number to shift (integer)
# @params[out] result : shifted array (complex array)
##


def shift_relief(arr, num):
    result = np.zeros_like(arr)
    if num > 0:  # upward
        # result[:num] = fill_value
        result[num:] = arr[:-num]

    elif num < 0:  # downward
        # result[num:] = fill_value
        result[:num] = arr[-num:]

    else:
        result[:] = arr
    return result
