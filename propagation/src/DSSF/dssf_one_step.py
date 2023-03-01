# This file is part of SSW-2D.
# SSW-2D is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# SSW-2D is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
# of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Foobar. If not, see
# <https://www.gnu.org/licenses/>.


#################################################################################################
# function : DSSF_one_step
# author : R. Douvenot
# Date : 04/12/18 / last modif : 24/02/2023
# State : OK
#
# function that compute one step of the DSSF for a field u
# psi_x_dx = DSSF_one_step(psi_x,simulation_parameters)
#
# The field is put in spectral domain with the spectral transform. Then the spectral potential
# is propagated using the diagonal propagator in Fourier domain. Last we come back in spatial domain
# with the inverse spectral transform.
#
#
# INPUTS :
# - u_x : (N_z) - array. potential at the x position
# - propagator : (N_z). Diagonal propagator
#
# OUTPUT :
# - u_x_dx : (N_z) - array. potential propagated to the x + dx position
#
#################################################################################################

import numpy as np
import time
from scipy.fft import fft, ifft, fftshift, ifftshift, dct, dst, idct, idst


def dssf_one_step(u_x, propagator):

    n_z = u_x.size

    # --- spectral transform --- #
    # t_start = time.process_time()
    u_kz_x = fftshift(fft(u_x, norm='ortho'))
    # t_end = time.process_time()
    # print('Transform time',t_end-t_start)

    # --- free-space propagation (spectrum) --- #
    # Propagation in spectral domain along x
    # t_start = time.process_time()
    u_kz_x_dx = u_kz_x * propagator
    # t_end = time.process_time()
    # print('Propa DSSF time one step',t_end-t_start)

    # --- inverse spectral transform --- #
    # t_start = time.process_time()
    u_x_dx = ifft(ifftshift(u_kz_x_dx), norm='ortho')
    # t_end = time.process_time()
    # print('Inverse time',t_end-t_start)

    return u_x_dx


#################################################################################################
# function : dssf_one_step_sin
# author : R. Douvenot
# Date : 24/02/23
# State : In progress
#
# function that compute one step of the DSSF for a field u above a ground with -1 reflection coefficient
# psi_x_dx = DSSF_one_step(psi_x,simulation_parameters)
#
# The field is put in spectral domain with the spectral transform. Then the spectral field
# is propagated using the diagonal propagator in Fourier domain. Last we come back in spatial domain
# with the inverse spectral transform.
#
#
# INPUTS :
# - u_x : (N_z) - array. field at the x position
# - simulation_parameters : Structure. Class containing the simulation parameters
#
# OUTPUT :
# - u_x_dx : (N_z) - array. field propagated to the x + dx position
#
#
#################################################################################################
def dssf_one_step_sin(u_x, propagator):

    n_z = u_x.size

    # DST (sinus transform) to go in the spectral domain
    f_u_x = dst(u_x, type=1, norm='ortho')

    # propagation of the spectrum
    f_u_x_dx = f_u_x * propagator

    # back in the space domain (without first and last points)
    u_x_dx = idst(f_u_x_dx, type=1, norm='ortho')

    return u_x_dx


#################################################################################################
# function : dssf_one_step_cos
# author : R. Douvenot
# Date : 24/02/23
# State : In progress
#
# function that compute one step of the DSSF for a field u above a ground with +1 reflection coefficient
# psi_x_dx = DSSF_one_step(psi_x,simulation_parameters)
#
# The field is put in spectral domain with the spectral transform. Then the spectral field
# is propagated using the diagonal propagator in Fourier domain. Last we come back in spatial domain
# with the inverse spectral transform.
#
#
# INPUTS :
# - u_x : (N_z) - array. field at the x position
# - propagator : Structure. Class containing the simulation parameters
#
# OUTPUT :
# - u_x_dx : (N_z) - array. field propagated to the x + dx position
#
#
#################################################################################################
def dssf_one_step_cos(u_x, propagator):
    n_z = u_x.size

    # DST (sinus transform) to go in the spectral domain
    f_u_x = dct(u_x, norm='ortho')

    # propagation of the spectrum
    f_u_x_dx = f_u_x * propagator

    # back in the space domain (without first and last points)
    u_x_dx = idct(f_u_x_dx, norm='ortho')

    return u_x_dx
