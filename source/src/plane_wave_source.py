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
# @package complex_source_point
# @brief Function that calculates the field radiated from a complex source point.
# @author T. Bonnafont
# @author R. Douvenot
# @version 1.0
# @date 19/07/21 (created)/ last modif : 07/04/21
# @details Computes the field radiated from a complex source point in 2D. Normalised by inputted radiated power
# and max gain. \n
# Source: G.A. Deschamps, Gaussian beam as a bundle of complex rays, Electron. Lett. 7 (1971) 684–685,
# https://doi.org/10.1049/el:19710467.
# @param[in] Geom   Structure that contains \n .Nz = nb of points along z (vertical) \n
#                                   .z_step = computation step along z (vertical)
# @param[in] csp    Structure that contains \n
#                                   .k0 = wavenumber in rad/m (real)  \n
#                                   .x_s = position of the csp along x (real <0) supposedly placed before x=0.  \n
#                                   .z_s = position of the csp along z (real) \n
#                                   .W0 = width of the complex source point (real) \n
#                                   .P_Tx = radiated power in W (real) \n
#                                   .G_Tx = max gain in dBi (real)
# @param[out] e_field   (n_z)-complex array. Contains the field radiated by the csp.
# Max norm = corresponds to the inputted power and max gain.
##


# libraries
import numpy as np
import scipy.constants as cst


# function complex_source_point
def plane_wave_source(config_source):

    # print('plane wave source at x_s = ' + np.str(config_source.x_s) + ' m')

    # --- GEOMETRY PARAMETERS --- #
    # k0 = wavenumber
    # number of points for the field array...
    n_z = config_source.n_z
    # ...and corresponding steps
    #z_step = config_source.z_step
    # --------------------------- #

    # --- SOURCE PARAMETERS --- #
    # frequency (through wavenumber)
    k0 = config_source.k0
    # vacuum impedance
    zeta0 = np.sqrt(cst.mu_0/cst.epsilon_0)
    # Real position
    x_s = config_source.x_s
    #z_s = config_source.z_s
    # with of the csp
    #w0 = config_source.W0
    # imaginary position of the source along x
    #xw0 = k0*(w0**2)/2.0
    # ------------------------- #

    # -------------------------------- #
    # --- Computation of the field --- #
    # -------------------------------- #

    # compute the complex x_position from the real position and the width
    #x_pos = -x_s + 1j*xw0

    # compute the z_pos from the source position
    z_vect = z_step * np.arange(0, n_z)


    #e_field = np.ones(n_z)*np.exp(-1j * k0 * x_s)
    #e_field = np.ones(n_z) * np.exp(1j * k0)
    e_field = np.ones(n_z)

    # ------------ END --------------- #
    # --- Computation of the field --- #
    # -------------------------------- #

    # --- NORMALISATION --- #
    normalisation_factor = np.max(np.abs(e_field))
    # max field amplitude = 1
    e_field /= normalisation_factor
    # max amplitude corresponds to the inputted gain and radiated power
    g_lin = 10**(config_source.G_Tx/10)
    e_field *= np.sqrt(zeta0 * config_source.P_Tx * g_lin / (2*np.pi)) / (-x_s)


    # --------------------- #

    return e_field
