# This file is part of SSW-2D.
# SSW-2D is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# SSW-2D is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
# of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Foobar. If not, see
# <https://www.gnu.org/licenses/>.


import numpy as np
import scipy.constants as cst
# import time


##
# @package u2w
# @author Remi Douvenot
# @date 16/05/2022
# @brief Calculates the change of variable from u to w to apply the DMFT
##

def u2w(alpha, u_field, n_z, z_step):
    # initialisation
    w_field = np.zeros(n_z, dtype=np.complex)  # index from 1 to n_z-2
    # gradient function does ( u[n+1] - u[n-1] ) / 2
    # w_field[1:n_z - 1] = np.gradient(u_field_2d)[1:n_z - 1] / z_step + alpha * u_field_2d[1:n_z - 1]
    w_field[1:n_z - 1] = (u_field[2:n_z] - u_field[0:n_z - 2]) / (2*z_step) + alpha * u_field[1:n_z - 1]

    return w_field


##
# @package u2w
# @author Remi Douvenot
# @date 16/05/2022
# @brief Calculates the change of variable from w to u to apply the inverse DMFT
##

def w2u(spectre_w_0, spectre_w_n_z, w_field, n_z, z_step, r0, aa):

    # initialisation of the variables
    yy = np.zeros(n_z, dtype=np.complex)  # intermediate variables for diff equation (from w to u)
    fp_field = np.zeros(n_z, dtype=np.complex)  # intermediate variables for diff equation (from w to u)

    # ----------------------------------------------------------------------- #
    # --- calculate fp_field the particular solution of the diff equation --- #
    # ----------------------------------------------------------------------- #
    # LU method
    yy[0] = 0 + 0j  # index = 0
    for jj_z in np.arange(1, n_z):  # index from 1 to n_z-1
        yy[jj_z] = r0 * yy[jj_z - 1] + 2.0 * w_field[jj_z] * z_step

    fp_field[n_z-1] = 0 + 0j  # index = n_z-1
    for jj_z in np.arange(n_z-2, -1, -1):  # index from n_z-2 to 0
        fp_field[jj_z] = r0 * (yy[jj_z] - fp_field[jj_z + 1])

    # ------------------------------------------------------------------------------ #
    # --- calculate B_1 and b_2 -> sum with first and last terms weighted by 1/2 --- #
    # ------------------------------------------------------------------------------ #

    if np.isnan(np.real(aa)) or np.isnan(np.imag(aa)):  # THEN
        # calculate B1 and B2
        b_1 = spectre_w_0
        b_2 = spectre_w_n_z
        print("NaN b_1 b_2")
    else:
        # Dockery (1996) eq.(29)-(30). 1st and last terms of the sum weighting = 0.5
        sum_temp1 = (fp_field[0] + fp_field[n_z-1] * r0**(n_z-1)) / 2.0
        sum_temp2 = (fp_field[n_z-1] + fp_field[0] * (-r0) ** (n_z-1)) / 2.0

        # other terms. ii_z in 1,n_z-1, weighting = 1
        sum_list1 = fp_field[1:n_z-1] * (r0 ** np.arange(1, n_z-1))
        sum_list2 = fp_field[n_z - 2:0:-1] * ((-r0) ** np.arange(1, n_z-1))

        # total sums
        sum_temp1 += np.sum(sum_list1)
        sum_temp2 += np.sum(sum_list2)

        # calculate B1 and B2
        b_1 = spectre_w_0 - aa * sum_temp1
        b_2 = spectre_w_n_z - aa * sum_temp2

    # retrieve u_field wrt b_1, b_2, and r0
    u_field = fp_field + b_1 * (r0 ** np.arange(0, n_z)) + b_2 * ((-r0) ** np.arange(n_z-1, -1, -1))
    # print('b1', b_1, 'b2', b_2)

    return u_field


##
# @package dmft_parameters
# @author Remi Douvenot
# @date 11/05/2022
# @brief Calculates the parameters for DMFT
##

def dmft_parameters(ii_x, config):
    # distance of calculation from the source --> for grazing angle
    dist_x = (ii_x * config.x_step) - config.x_s
    # angle of incidence at distance dist_x --> used for wavenumbers and Fresnel coefficient
    theta_inc = np.arctan(dist_x/config.z_s)
    # wavenumbers
    [k0, k_2, k_iz, k_tz] = calculate_wavenumbers(config, theta_inc)
    r_fresnel = fresnel_coefficient(1.0, config.epsr, k_iz, k_tz, config.polar)

    # The DMFT parameters themselves: alpha, r0 and A
    # alpha
    alpha = -1j * k0 * np.cos(theta_inc) * (1.0 - r_fresnel) / (1.0 + r_fresnel)
    # r0
    r0 = calculate_r0(config.z_step, alpha, config.polar)
    # AA
    aa = 2.0*(1.0-r0**2) / ((1.0+r0**2) * (1.0-r0**(2*config.N_z)))

    # print('alpha = ', alpha)
    # print('r0 = ', r0)
    # print('aa = ', aa)

    return alpha, r0, aa


##
# @package calculate_wavenumbers
# @author Remi Douvenot
# @date 11/05/2022
# @brief Calculates the wavenumbers for Fresnel coefficient
##

def calculate_wavenumbers(config, theta_inc):

    freq0 = config.freq
    epsr = config.epsr  # second medium = ground

    k0 = 2.0*np.pi*freq0/cst.c     # wavenumber in medium 1 (= air)
    ksol = 2.0*np.pi*freq0/cst.c * np.sqrt(epsr)   # wavenumber in medium 2
    k_iz = - k0*np.cos(theta_inc)  # incident wavenumber along z
    k_tx = k0*np.sin(theta_inc)   # incident/transmitted wavenumber along x (k_tx = k_ix)
    k_tz = np.sqrt(ksol**2-k_tx**2)  # transmitted wavenumber along z. sqrt with positive imag part
    if np.imag(k_tz) < 0:
        k_tz = - k_tz
    return k0, ksol, k_iz, k_tz


##
# @package fresnel_coefficient
# @author Remi Douvenot
# @date 12/05/2022
# @brief Calculates the Fresnel coefficient
# @param[in] epsr_1: relative permittivity in medium 1 (complex)
# @param[in] epsr_2: relative permittivity in medium 2 (complex)
# @param[in] k_iz: vertical wavenumber in medium 1 (real)
# @param[in] k_tz: vertical wavenumber in medium 2 (real)
# @param[in] polarisation: can be 'TE' or 'TM'
# @param[out] r_fresnel: the reflection coefficient (complex)
##


def fresnel_coefficient(epsr_1, epsr_2, k_iz, k_tz, polarisation):
    if polarisation == 'TM':
        r_fresnel = (epsr_2*k_iz - epsr_1*k_tz) / (epsr_2*k_iz + epsr_1*k_tz)
    elif polarisation == 'TE':
        r_fresnel = (k_iz - k_tz) / (k_iz + k_tz)
    else:
        raise ValueError(['Unknown polarisation. Polarisation should be ''TE'' or ''TM'''])
    return r_fresnel


##
# @package calculate_r0
# @author Remi Douvenot
# @date 12/05/2022
# @brief Calculates the parameter r0 used in the DMFT transform (discrete mixed Fourier transform)
# @param[in] z_step: vertical step (m)
# @param[in] alpha: another DMFT parameter previously calculated
# @param[out] r0: the r0 parameter. Root of r^2 + 2 r alpha z_step -1 = 0 Levy 1st ed. eq(9.52)
##


def calculate_r0(z_step, alpha, polarisation):
    if polarisation == 'TM':  # TM polarisation
        # alpha has a positive real part
        r0 = np.sqrt(1+(alpha*z_step)**2) - alpha*z_step
    else:  # TE polarisation
        # alpha has a negative real part
        r0 = - np.sqrt(1+(alpha*z_step)**2) - alpha*z_step
    return r0


##
# @package surface_wave_propagation
# @author Remi Douvenot
# @date 16/05/2022
# @brief Calculates propagation of the surface wave terms in the DMFT spectrum
# @param[in] w_field: field under its w form
# @param[in] config: class containing the simulation settings
# @param[in] alpha: another DMFT parameter previously calculated
# @param[out] r0: the r0 parameter. Root of r^2 + 2 r alpha z_step -1 = 0 Levy 1st ed. eq(9.52)
##

def surface_wave_propagation(w_field, config, r0, aa):

    n_z = config.N_z
    x_step = config.x_step
    z_step = config.z_step
    lambda0 = cst.c/config.freq
    k0 = 2*cst.pi / lambda0

    # ------------------------------------------------------------------------------------ #
    # --- calculate W(0) and W(N_z-1) -> sum with first and last terms weighted by 1/2 --- #
    # ------------------------------------------------------------------------------------ #

    if np.isnan(np.real(aa)) or np.isnan(np.imag(aa)) or config.freq > 1e9:
        spectrum_w_0 = 0
        # print('No surface wave')
    else:
        # first terms. ii_z = 0, weighting = 1 / 2 (initialisation of the sum)
        spectrum_w_0 = w_field[0] / 2.0
        # other terms. ii_z in 1,NN_z-1, weighting = 1
        sum_list = w_field[1:n_z] * r0**np.arange(1, n_z)
        spectrum_w_0 += aa * np.sum(sum_list)
        # last terms. ii_z = NN_z, weighting = 1/2
        spectrum_w_0 += aa / 2.0 * (r0 ** (n_z-1)) * w_field[n_z-1]

        # -------------------------------------- #
        # --- Propagation in spectral domain --- #
        # -------------------------------------- #

        spectrum_w_0 = spectrum_w_0 * np.exp(-((1j * x_step) / (2.0 * k0)) * ((np.log(r0) / z_step) ** 2))

    # last value put at 0 following Dockery, 1996
    spectrum_w_n_z = 0.0

    return spectrum_w_0, spectrum_w_n_z
