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
# @package apply_apodisation
# @author Remi Douvenot
# @brief apply the vertical and horizontal apodisation windows
# @warning apodisation type Hanning is the only one coded
##

import numpy as np
import time
from src.wavelets.wavelet_operations import q_max_calculation


def apply_apodisation(u_x, apo_window_z, config):

    # size of the apodisation windows
    n_apo_z = apo_window_z.size

    # is there a ground?
    if config.ground == 'None':
        # apply apodisation along z (top and bottom)
        u_x[-n_apo_z:] *= apo_window_z
        u_x[:n_apo_z] *= apo_window_z[::-1]
    else:
        # apply apodisation along z (top only)
        u_x[-n_apo_z:] *= apo_window_z

    return u_x

##
# @package apodisation_window
# @brief Create a 2D apodisation window
# @warning apodisation type Hanning is the only one coded
##


def apodisation_window(apo_window, n_apo):

    # Hanning window
    if apo_window == 'Hanning':
        # compute the Hanning window on N_apo points
        apo_vect = np.arange(0, n_apo)
        apo_window = (1.0 + np.cos(np.pi * apo_vect / n_apo)) / 2.0
    else:
        raise 'ERROR : NOT CODED YET'

    return apo_window

##
# @package apply_apodisation_wavelet
# @author Remi Douvenot
# @brief apply the vertical apodisation window on the wavelet coefficients directly
# @warning apodisation type Hanning is the only one coded
# @warning does nothing yet
##


def apply_apodisation_wavelet(w_x, apo_window_z, config):

    # number of q_max per level
    q_max = q_max_calculation(config.wv_L)
    # decimation coefficient per level
    decimation = (2**config.wv_L/q_max).astype(int)

    # size of the apodisation windows
    n_apo_z = apo_window_z.size

    # apodisation on each level
    for ii_l in np.arange(0, config.wv_L + 1):
        w_x_ll = w_x[ii_l]
        delta = decimation[ii_l]
        n_apo_z_delta = int(n_apo_z/delta)

        # apply apodisation along z (top of the vector)
        w_x_ll[-n_apo_z_delta:] *= apo_window_z[::delta]
        # apply apodisation along z (bottom)
        if config.ground == 'None':
            w_x_ll[:n_apo_z_delta] *= apo_window_z[::-delta]
        w_x[ii_l] = w_x_ll

    return w_x
