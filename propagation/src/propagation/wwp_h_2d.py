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
# @file wwp_h_2d.py
#
# @author R. Douvenot
# @package WWP_H_2D
# @date 18/11/22
# @version In progress
# @brief The core of the 2D WWP-H: propagates a 1D initial field with hybrid SSW-WWP in 2D.
# @details This function is the core of the 2D WWP-H code. It propagates a 1D initial reduced field with an
# hybridization of WWP and SSW in 2D, in a Cartesian environment. The output field is given in the wavelet domain for
# a reduced memory size. \n
# The library of propagators is computed. Then the field is propagated in the wavelet domain using this library. \n
# The refractivity is taken into account with a phase screen applied on wavelets directly.
# Same for apodisation. \n
# The ground is accounted on the bottom of the domain using SSW.
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

# u_x_dx = SSW(u_0,simulation_parameters)
#
#######################################################################################################################

import numpy as np
import time
import scipy.constants as cst
from src.propagators.dictionary_generation import dictionary_generation
from src.propagation.wwp_2d_one_step import wwp_2d_one_step
from src.propagation.ssw_2d_one_step import ssw_2d_one_step
from src.propagation.apodisation import apply_apodisation, apply_apodisation_wavelet, apodisation_window
from src.DSSF.dmft import dmft_parameters, u2w, w2u, surface_wave_propagation
import pywt
from src.wavelets.wavelet_operations import thresholding
from src.propagation.image_field import compute_image_field, compute_image_field_tm_pec
from src.propagation.refraction import apply_refractive_index_wavelet, apply_refractive_index
from src.wavelets.wavelet_operations import sparsify, q_max_calculation, hybrid_ssw_wwp, extract_ssw, remove_image_coef
from src.wavelets.wavelet_operations import disassemble_ssw_wwp, assemble_ssw_wwp
from src.propagation.ssw_2d_one_step import wavelet_propag_one_step


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
# @profile


def wwp_h_2d(u_0, config, n_refraction, ii_vect_relief):

    # Simulation parameters
    n_x = config.N_x

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
    np.save('./outputs/dictionary', dictionary)
    # --- Relief and Sizes of the apodisation and image layers --- #
    if config.ground == 'PEC' or config.ground == 'Dielectric':
        # derivative of the relief calculated once for all
        diff_relief = np.diff(ii_vect_relief)
        # size of the image layer must be a multiple of 2**L
        n_im = int(np.round(config.N_z * config.image_layer))
        remain_im = n_im % 2**config.wv_L
        if remain_im != 0:
            n_im += 2**config.wv_L - remain_im
    else:  # config.ground == 'None':
        print('--- Main loop. No ground ---')
        n_im = 0
    config.N_im = n_im

    n_apo_z = int(config.apo_z * config.N_z)
    remain_apo = n_apo_z % 2 ** config.wv_L
    if remain_apo != 0:
        n_apo_z += 2 ** config.wv_L - remain_apo
    # ------------------------------------------------- #

    # --- Creation of the apodisation window --- # @todo Code other apodisation windows
    # along z
    apo_window_z = apodisation_window(config.apo_window, n_apo_z)
    # ------------------------------------------ #

    # --- Initialisations --- #
    # lengths of the field vectors in the WWP and in the SSW domains
    n_z = config.N_z
    # initial field
    u_0 = apply_apodisation(u_0, apo_window_z, config)
    # field on the bottom domain : n_im of electric field + n_im of hybridisation (+n_im for image later)
    u_ssw_x = u_0[0: 2*n_im]
    # Initialise the wavelet vectors in the WWP and SSW domains
    w_wwp_x = pywt.wavedec(u_0, config.wv_family, 'per', config.wv_L)
    w_wwp_x = thresholding(w_wwp_x, config.V_s)
    # remove the size image layer corresponding to SSW
    w_wwp_x = remove_image_coef(w_wwp_x, config)
    # Initialise (zeros) the current wavelet decomposition
    u_x = np.zeros(n_z + n_im, dtype=complex)
    w_x = pywt.wavedec(u_x, config.wv_family, 'per', config.wv_L)

    # saved total wavelet parameters (sparse coo matrix)
    wv_total = [[]] * n_x
    # ----------------------- #

    # Loop over the x_axis
    for ii_x in np.arange(1, n_x+1):
        if ii_x % 100 == 0:
            print('Iteration', ii_x, '/', n_x, '. Distance =', ii_x*config.x_step)

        # ------------------------------ #
        # --- Free-space propagation --- #
        # ------------------------------ #
        if config.ground == 'Dielectric':

            raise ValueError(['Dielectric ground not yet available in WWP-H'])

        elif config.ground == 'PEC':

            # raise ValueError(['PEC ground not yet available in WWP-H'])

            # descending relief
            if diff_relief[ii_x - 1] < 0:
                # Add zeros below the field
                raise ValueError(['Relief not yet available in WWP-H'])
                # u_x = shift_relief(u_x, -diff_relief[ii_x - 1])

            # Add the image layer to apply the local image method #
            if config.polar == 'TE':
                u_ssw_x = compute_image_field(u_ssw_x, n_im)
            elif config.polar == 'TM':
                u_ssw_x = compute_image_field_tm_pec(u_ssw_x, n_im)

            # Wavelet parameters
            family = config.wv_family
            mode = 'per'

            # Decompose the SSW field into a wavelet coefficient vector
            w_ssw_x = pywt.wavedec(u_ssw_x, family, mode, config.wv_L)
            # Threshold V_s on the signal
            w_ssw_x = thresholding(w_ssw_x, config.V_s)
            # # Hybridization with WWP = duplicate the coefficients at the top of SSW coefficients
            # w_ssw_x = hybrid_ssw_wwp(w_ssw_x, w_wwp_x, config)

            # create the total wavelet coefficients
            w_x = assemble_ssw_wwp(w_x, w_ssw_x, w_wwp_x, config)
            w_x = thresholding(w_x, config.V_s)
            # apply apodisation on the wavelets
            w_x = apply_apodisation_wavelet(w_x, apo_window_z, config)
            # config.N_z = n_ssw
            w_x_dx = wavelet_propag_one_step(w_x, dictionary, config)
            # Threshold V_s on the signal
            w_x_dx = thresholding(w_x_dx, config.V_s)

            # ascending relief
            if diff_relief[ii_x - 1] > 0:
                # Put zeros in the relief
                raise ValueError(['Relief not yet available in WWP-H'])
                # u_x_dx = shift_relief(u_x_dx, -diff_relief[ii_x - 1])
                # u_x_dx[0:diff_relief[ii_x - 1]] = 0.0

            # # Propagate using WWP
            # config.N_z = n_wwp
            # config.N_im = 0
            # w_wwp_x_dx = wwp_2d_one_step(w_wwp_x, dictionary, config)

            # # recompose the wavelets = stick together ssw and wwp and remove image layer
            # config.N_z = n_z
            # config.N_im = n_im
            # w_x_dx = assemble_ssw_wwp(w_x, w_ssw_x_dx, w_wwp_x_dx, config)

        elif config.ground == 'None':

            raise ValueError(['If no ground, please use WWP instead of WWP-H'])

        else:
            raise ValueError(['Ground condition should be dielectric, PEC, or None'])
        # ---------- END --------------- #
        # --- Free-space propagation --- #
        # ------------------------------ #

        # extract SSW and WWP parts for total WWP-H decomposition
        w_ssw_x_dx, w_wwp_x_dx = disassemble_ssw_wwp(w_x_dx, w_ssw_x, w_wwp_x, config)
        # SSW field using inverse wavelet transform
        u_ssw_x_dx = pywt.waverec(w_ssw_x_dx, family, mode)
        # remove image field
        u_ssw_x_dx = u_ssw_x_dx[n_im:]

        # --- refractivity applied on WWP and SSW separately --- #
        # lower atm = SSW
        u_ssw_x_dx = apply_refractive_index(u_ssw_x_dx, n_refraction[0:2*n_im], config)
        u_ssw_x_dx = apply_refractive_index(u_ssw_x_dx, n_refraction[0:2*n_im], config)
        # upper atm = WWP
        w_wwp_x_dx = apply_refractive_index_wavelet(w_wwp_x_dx, n_refraction[n_im:], config)
        # ------------------------------------------------------ #

        # store the wavelet parameters (in coo format)
        # w_x_dx = remove_image_coef(w_x_dx, config)
        wv_total[ii_x-1] = sparsify(w_x_dx)
        # store the wavelet parameters (in coo format)
        # spectrum_w_0_tot[ii_x - 1] = spectrum_w_0_tot

        # update
        # w_wwp_x = w_wwp_x_dx
        u_ssw_x = u_ssw_x_dx
        w_wwp_x = w_wwp_x_dx

    # last field is saved in the space domain
    u_x_dx = pywt.waverec(w_x_dx, config.wv_family, mode='per')

    return u_x_dx, wv_total


