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
# @file wwp_2d.py
#
# @author R. Douvenot
# @package WWP_2D
# @date 08/11/22
# @version In progress
# @brief The core of the 2D WWP: propagates a 1D initial field with WWP in 2D.
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

# u_x_dx = SSW(u_0,simulation_parameters)
#
#######################################################################################################################

import numpy as np
import time
import scipy.constants as cst
from src.propagators.dictionary_generation import dictionary_generation
from src.propagation.wwp_2d_one_step import wwp_2d_one_step
from src.propagation.apodisation import apply_apodisation, apply_apodisation_wavelet, apodisation_window
from src.DSSF.dmft import dmft_parameters, u2w, w2u, surface_wave_propagation
import pywt
from src.wavelets.wavelet_operations import thresholding, q_max_calculation
from src.propagation.refraction import apply_refractive_index_wavelet
from src.wavelets.wavelet_operations import sparsify  # for sparsify


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


def wwp_2d(u_0, config, n_refraction):

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
    # initial field
    u_0 = apply_apodisation(u_0, apo_window_z, config)
    # Decompose the initial field into a wavelet coefficient vector
    w_x = pywt.wavedec(u_0, config.wv_family, 'per', config.wv_L)
    # Threshold V_s on the signal
    w_x = thresholding(w_x, config.V_s)

    # saved total wavelet parameters (sparse coo matrix)
    wv_total = [[]] * n_x
    # ----------------------- #

    # Loop over the x_axis
    for ii_x in np.arange(1, n_x+1):
        if ii_x % 100 == 0:
            print('Iteration', ii_x, '/', n_x, '. Distance =', ii_x*config.x_step)
        # --- apodisation applied on wavelets --- #
        w_x = apply_apodisation_wavelet(w_x, apo_window_z, config)
        # --------------------------------------- #

        # ------------------------------ #
        # --- Free-space propagation --- #
        # ------------------------------ #
        if config.ground == 'Dielectric':

            raise ValueError(['Dielectric ground not yet available in WWP'])

        elif config.ground == 'PEC':

            raise ValueError(['PEC ground not yet available in WWP'])

        elif config.ground == 'None':

            # Propagate using WWP
            w_x_dx = wwp_2d_one_step(w_x, dictionary, config)

        else:
            raise ValueError(['Ground condition should be dielectric, PEC, or None'])
        # ---------- END --------------- #
        # --- Free-space propagation --- #
        # ------------------------------ #

        # --- refractivity applied on wavelets --- #
        w_x_dx = apply_refractive_index_wavelet(w_x_dx, n_refraction, config)
        # ------------------------------__-------- #

        # store the wavelet parameters (in coo format)
        wv_total[ii_x-1] = sparsify(w_x_dx)
        # store the wavelet parameters (in coo format)
        # spectrum_w_0_tot[ii_x - 1] = spectrum_w_0_tot
        # update w_x
        w_x = w_x_dx

    # back from wavelet to field
    u_x_dx = pywt.waverec(w_x_dx, config.wv_family, mode='per')

    return u_x_dx, wv_total

