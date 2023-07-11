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
# @package SSW_2D_LIGHT
# @date 11/07/2023
# @version Test for JEREMY
# @brief The core of the 2D SSW: propagates a 1D initial field with SSW in 2D.
# @details This function is the core of the 2D SSW code. It propagates a 1D initial reduced field with SSW in 2D,
# in a Cartesian environment. The output field is given in the wavelet domain for a reduced memory size. \n
# The library of propagators is computed. Then the field is propagated in the wavelet domain using this library. \n
# Refractivity and Apodisation ARE REMOVED
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
from src.atmosphere.genere_n_profile import genere_phi_turbulent
from src.DSSF.dmft import dmft_parameters, u2w, w2u, surface_wave_propagation
from src.wavelets.wavelet_operations import sparsify, q_max_calculation


def ssw_2d_light(u_0, config):

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

    # ----------------------- #
    # --- Library reshape --- #
    # ----------------------- #
    # The propagator LIBRARY is per default in a "pywavelet" shape
    # These lines put it back in a vector shape
    # TODO : tip for JEREMY : maybe make this step befor and put the dictionary as an input of SSW_2D_cpp
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
    # --------- END --------- #
    # --- Library reshape --- #
    # ----------------------- #

    # --- Initialisations --- #
    # initial field
    u_x = u_0
    # field after delta x
    u_x_dx = np.zeros_like(u_x)

    # Loop over the x_axis
    for ii_x in np.arange(1, n_x+1):
        # if you want to display ii_x
        # if ii_x % 100 == 0:
        #     print('Iteration', ii_x, '/', n_x, '. Distance =', ii_x*config.x_step)

        # -------------------------------------- #

        # ------------------------------ #
        # --- Free-space propagation --- #
        # ------------------------------ #

        # Propagate using SSW
        u_x_dx, wavelets_x_dx = ssw_2d_one_step(u_x, dictionary, n_propa_lib, config)

        # ---------- END --------------- #
        # --- Free-space propagation --- #
        # ------------------------------ #

        u_x = u_x_dx

    return u_x_dx
