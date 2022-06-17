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
