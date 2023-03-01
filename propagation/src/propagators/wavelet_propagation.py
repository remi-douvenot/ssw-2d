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
# @file wavelet_propagation.py
#
# @author T. Bonnafont
# @brief Comment to write!
##

from src.DSSF.dssf_one_step import dssf_one_step
from src.DSSF.propa_discrete_spectral_domain import discrete_spectral_propagator
import numpy as np
import math
import matplotlib.pyplot as plt
import sys


def wavelet_propagation(u_wavelet_x, config):

    n_z_tot = u_wavelet_x.size
    # N_y_pre,N_z_pre = P_DSSF.shape
    # if N_y_tot != N_y_pre or N_z_tot!=N_z_pre:
    propagator_dssf = discrete_spectral_propagator(config, n_z_tot)

    u_wavelet_x_dx = dssf_one_step(u_wavelet_x, propagator_dssf)

    return u_wavelet_x_dx
