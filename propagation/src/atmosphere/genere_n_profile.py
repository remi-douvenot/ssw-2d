##
# @brief Function that generates the vertical profile of n or m index
# @author R. Douvenot
# @package genere_n_profile.py
# @date 08/09/21
# @version Work in progress
#
# @details Function that generates the vertical profile of n or m index. Available atmospheres: homogeneous, standard,
# bilinear, trilinear, evaporation duct, double duct, loaded from a file.
# n_refraction = genere_n_profile(config)
#
# @params[in] config : class with the parameters
# @params[out] n_profile : vertical profile of n or m (modified) refractive index
##

import numpy as np
import matplotlib.pyplot as plt


def generate_n_profile(config):
    # choice of the atmospheric profile type
    if config.atmosphere == 'Homogeneous':  # constant n (vacuum)
        n_refractive_index = np.ones(config.N_z)
    elif config.atmosphere == 'Linear':  # constant slope
        n_refractive_index = linear_atmosphere(config.N_z, config.z_step, config.c0)
    elif config.atmosphere == 'Evaporation':  # log then constant slope
        n_refractive_index = evaporation_duct(config.N_z, config.z_step, config.c0, config.delta)
    elif config.atmosphere == 'Bilinear':  # bilinear profile = trilinear with zb = 0
        n_refractive_index = trilinear_profile(config.N_z, config.z_step, config.c0, 0.0, config.c2, config.zt)
    elif config.atmosphere == 'Trilinear':  # trilinear
        n_refractive_index = trilinear_profile(config.N_z, config.z_step, config.c0, config.zb, config.c2, config.zt)
    elif config.atmosphere == 'Double duct':  # log then trilinear
        n_refractive_index = double_duct(config.N_z, config.z_step, config.c0, config.delta, config.zb, config.c2, config.zt)
    elif config.atmosphere == 'file':  # complex profile from file
        n_refractive_index = read_file_profile(config.N_z, config.z_step, config.atm_filename)
    else:
        raise ValueError(['Wrong atmospheric type!'])

    return n_refractive_index


# standard atmosphere
def linear_atmosphere(n_z, z_step, c0):
    # vector of vertical positions
    z_vect = np.linspace(0, z_step*n_z, n_z, endpoint=False)
    # refractivity
    # mean M0 is 330 at the ground level
    n_refractivity = 330 + z_vect*c0
    # refractive index
    n_refractive_index = 1 + n_refractivity*1e-6
    return n_refractive_index


# evaporation duct
def evaporation_duct(n_z, z_step, c0, delta):
    # vector of vertical positions
    z_vect = np.linspace(0, z_step*n_z, n_z, endpoint=False)
    n_refractivity = np.zeros_like(z_vect)
    # z0
    z0 = 1.5e-4
    # separating in and above the duct (where renders tuples)
    indices_z_inf = np.where(z_vect <= 2*delta)
    indices_z_sup = np.where(z_vect > 2*delta)
    # in the duct: refractivity following Paulus-Jeske model
    n_refractivity[indices_z_inf] = 330 + 0.125*(z_vect[indices_z_inf] -
                                    delta*np.log((z_vect[indices_z_inf]+z0) / z0))
    # above the duct: standard atm
    n_refractivity[indices_z_sup] = 330 + z_vect[indices_z_sup]*c0
    # ensuring continuity
    n_refractivity[indices_z_inf] = n_refractivity[indices_z_inf] - n_refractivity[indices_z_inf[0][-1]] \
                                  + n_refractivity[indices_z_sup[0][0]] - c0*z_step
    # refractive index
    n_refractive_index = 1 + n_refractivity * 1e-6
    return n_refractive_index
# This file is part of SSW-2D.
# SSW-2D is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# SSW-2D is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
# of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Foobar. If not, see
# <https://www.gnu.org/licenses/>.

# # bilinear profile
# def bilinear_profile(n_z, z_step, c0, c2, zt):
#     raise ValueError(['bilinear_profile duct must be validated'])
#     # vector of vertical positions
#     z_vect = np.linspace(0, z_step*n_z, n_z, endpoint=False)
#     n_refractivity = np.zeros_like(z_vect)
#     # --- refractivity --- #
#     # mean M0 is 330 at the ground level
#     # in the duct
#     n_refractivity[z_vect <= zt] = 330 + z_vect[z_vect <= zt]*c2
#     # above the duct
#     n_refractivity[z_vect > zt] = 330 + zt*c2 + (z_vect[z_vect > zt]-zt) * c0
#     # --- refractive index --- #
#     n_refractive_index = 1 + n_refractivity*1e-6
#     return n_refractive_index


# trilinear profile
def trilinear_profile(n_z, z_step, c0, zb, c2, zt):
    # vector of vertical positions
    z_vect = np.linspace(0, z_step * n_z, n_z, endpoint=False)
    n_refractivity = np.zeros_like(z_vect)
    # z0
    z0 = 1.5e-4
    # print(indices_z_inf)
    # below the duct
    indices_zb = np.where(z_vect <= zb)  # below the duct
    n_refractivity[indices_zb] = 330 + z_vect[indices_zb] * c0
    # in the duct
    indices_zt = np.where(z_vect > zb)
    n_refractivity[indices_zt] = 330 + zb * c0 + (z_vect[indices_zt] - zb) * c2
    # above the duct
    indices_above = np.where(z_vect > zt + zb)  # above the duct
    n_refractivity[indices_above] = 330 + zb * c0 + zt * c2 + (z_vect[indices_above] - zb - zt) * c0
    # --- refractive index --- #
    n_refractive_index = 1 + n_refractivity*1e-6
    return n_refractive_index


# double duct
def double_duct(n_z, z_step, c0, delta, zb, c2, zt):
    raise ValueError(['double duct not yet coded'])
    n_refractive_index = 1
    return n_refractive_index


# read profile from a text file
def read_file_profile(n_z, z_step, atm_filename):
    raise ValueError(['read_file_profile not yet coded'])
    n_refractive_index = 1
    return n_refractive_index
