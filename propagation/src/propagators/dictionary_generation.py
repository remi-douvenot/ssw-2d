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
# @file dictionary_generation.py
#
# @package dictionary_generation
# @author Remi Douvenot
# @date 20/07/2021
# @brief Calculates the dictionary of the wavelet propagators
# @param[in] config Class with all the simulation parameters
# @param[out] dictionary Dictionary containing all the propagators
# @details Calculates the dictionary of the wavelet propagators for 2D propagation
##

import numpy as np
import time
import math
import pywt
from src.wavelets.wavelet_operations import thresholding, q_max_calculation
from src.propagators.wavelet_propagation import wavelet_propagation
import scipy
import matplotlib.pyplot as plt


def dictionary_generation(config):
    # Maximum level of the wavelet decomposition
    ll = config.wv_L

    # 1/ calculate the wavelets to propagate
    # they already have the proper size for the DSSF propagation
    u_wavelets = wavelet_fun_dictionary(config)

    # 2/ the set of propagators is the IFWT of each wavelet with the appropriate translations
    # Initialisation: Creation of an empty list with the proper size
    dictionary = init_dictionary(ll)
    # number of translations at each level
    q_max_list = q_max_calculation(ll)

    # ---------------------------------------------- #
    # --- Main loop. Fill each level recursively --- #
    # ---------------------------------------------- #
    for ii_lvl in np.arange(0, ll+1):
        # Normalised level
        # the translations needed on each direction q = 2^(ll+1-l) (theoretical) and q = 2^(l-1) in this code
        q_max = q_max_list[ii_lvl]

        # Propagate the centered wavelet using DSSF
        # DSSF propagation
        u_wv_lvl_p = wavelet_propagation(u_wavelets[ii_lvl], config)
        # u_wv_lvl_p = u_wavelets[ii_lvl]

        t_start = time.process_time()
        # print('size u_wavelet',u_wavelet_2D.shape)
        for q_z in np.arange(0, q_max):
            # Shift the wavelet in y and z directions
            u_wv_lvl_p_t = shift_field(u_wv_lvl_p, ii_lvl, ll, q_z)
            # Come back in the wavelet domain
            # t_start = time.process_time()
            propagator_il_in_q = pywt.wavedec(u_wv_lvl_p_t, config.wv_family, 'per', ll)
            # threshold of V_m applied on those coefficient
            propagator_il_in_q = thresholding(propagator_il_in_q, config.V_p*np.max(np.abs(u_wv_lvl_p_t)))
            # t_end = time.process_time()
            # print('transform time', t_end-t_start)
            # Save the sparse vector of (l,n,q_y,q_z)-wavelet at the position [ii_lvl][ii_or][q_y][q_z]
            dictionary[ii_lvl][q_z] = propagator_il_in_q
        # t_end = time.process_time()
        # print('Time to do every translation', t_end - t_start)
    return dictionary

    # --------------------- END -------------------- #
    # --- Main loop. Fill each level recursively --- #
    # ---------------------------------------------- #

##
# @package waveletfun_dictionary
# @brief generate the 1D wavelet dictionary of wavelets to propagate
# @author Remi Douvenot
# @date 26/04/2020
# @param[in] config Class that contains all the SSW configuration
# @warning Not tested for different steps (in Y and Z)
##


def wavelet_fun_dictionary(config):

    # max wavelet decomposition level
    ll = config.wv_L

    # --- Initialise the wavelet source as L+1 empty arrays --- #
    u_wavelets = [[] for _ in range(0, ll+1)]

    # --- add the points for DSSF propagation (WAPE validity domain) --- #
    # theoretical support of the wavelet (before propagation)
    n_top, n_bottom = wavelet_size(config.wv_family, 0)
    # theoretical support of the wavelet (after propagation)
    n_u_top, n_u_bottom = wavelet_propa_size(n_bottom, config.x_step, config.z_step, ll)
    # scaling function
    u_wavelets[0] = create_dssf_scaling_fct(n_u_top, n_u_bottom, config.wv_family, 1, ll)

    # generation of the wavelets
    # loop on the level
    for ii_lvl in np.arange(1, ll+1):
        # print('ii_lvl =', ii_lvl)
        # theoretical support of the wavelet (before propagation)
        n_top, n_bottom = wavelet_size(config.wv_family, ii_lvl)

        # vertical support after propagation
        n_u_top, n_u_bottom = wavelet_propa_size(n_bottom, config.x_step, config.z_step, ll)
        # vertical wavelet
        u_wavelets[ii_lvl] = create_dssf_wavelet(n_u_top, n_u_bottom, config.wv_family, ii_lvl, ll)

    return u_wavelets

##
# @package create_DSSF_wavelet
# @brief Put the wavelet on its appropriate size
# @author Remi Douvenot
# @date 04/05/2021
# @version V1
# @warning: explain inputs and outputs !!!
# @warning: improve the generation of the empty wavelet 2D decomposition
##


def create_dssf_wavelet(n_u_top, n_u_bottom, family, ii_lvl, ll):
    # family = wavelet family (sym6)
    # ii_lvl = wavelet level, 0 = scaling function
    # ll = max wavelet decomposition level
    # x_step = x_step of SSW (and DSSF)
    # other_step = z_step or y_step of SSW (and DSSF)

    if family != 'sym6':
        raise ValueError('Incorrect wavelet family. only sym6 is available')

    u_wavelet = np.zeros(n_u_top + n_u_bottom, dtype='complex')
    # 1/ generate an empty wavelet decomposition of the corresponding size
    wv_dec = pywt.wavedec(u_wavelet, family, 'per', ll)

    # 2/ median wavelet = 1
    ii_one = int(n_u_bottom / 2 ** (ll - ii_lvl + 1))
    wv_dec[ii_lvl][ii_one] = 1
    # tests to print the sizes of the wavelets
    '''print('In dictionary_generation.py: level =', np.str(ii_lvl))
    print('In dictionary_generation.py: size signal top = ', np.str(N_u_top), 'size signal bottom = ', np.str(N_u_bottom), 'size signal = ', np.str(N_u_top+N_u_bottom))
    print('In dictionary_generation.py: size wavelet = ', np.str(wv_dec[ii_lvl].size), ' position = ', ii_one)'''

    # 3/ wavelet signal = IFWT of this wavelet decomposition
    u_wavelet_out = pywt.waverec(wv_dec, family, mode='per')
    return u_wavelet_out

##
# @package create_DSSF_scaling_fct
# @brief Put the wavelet on its appropriate size
# @author Remi Douvenot
# @date 04/05/2021
# @brief Calculate a scaling function for dictionary
# @warning Should be removed by using create_DSSF_wavelet only
##


def create_dssf_scaling_fct(n_u_top, n_u_bottom, family, ii_lvl, ll):
    # N_u_wavelet: support of the wavelet after propa
    # family = wavelet family (sym6)
    # ii_lvl = wavelet level, 0 = scaling function
    # ll = max wavelet decomposition level

    if family != 'sym6':
        raise ValueError('Incorrect wavelet family. only sym6 is available')

    # 1/ generate an empty wavelet decomposition of the corresponding size
    u_wavelet = np.zeros(n_u_top + n_u_bottom, dtype='complex')
    wv_dec = pywt.wavedec(u_wavelet, family, 'per', ll - ii_lvl + 1)

    # 2/ median wavelet = 1
    ii_one = int(n_u_bottom / 2 ** (ll - ii_lvl + 1))
    wv_dec[0][ii_one] = 1

    # tests to print the sizes of the wavelets
    '''print('In dictionary_generation.py: level =', np.str(ii_lvl))
    print('In dictionary_generation.py: size signal top = ', np.str(N_u_top), 'size signal bottom = ', np.str(N_u_bottom), 'size signal = ', np.str(N_u_top+N_u_bottom))
    print('In dictionary_generation.py: size wavelet = ', np.str(wv_dec[0].size), ' position = ', ii_one)'''

    # 3/ wavelet signal = IFWT of this wavelet decomposition
    u_wavelet_out = pywt.waverec(wv_dec, family, mode='per')

    return u_wavelet_out


##
# @package wavelet_size
# @brief Calculate the size of the wavelet before propagation
# @author Remi Douvenot
# @date 20/07/2021
# @version V1
# @param[in] family: wavelet family ("sym6")
# @param[in] ii_lvl: current wavelet level
# @param[out] N0_top: number of points on top part of the wavelet
# @param[out] N0_bottom: number of points at the bottom part of the wavelet
# @warning Only "sym6" coded
##


def wavelet_size(family, ii_lvl):

    # Original upper and lower sizes of the smallest wavelet of the family.
    if family == 'sym6':
        # the 12-sized wavelet has 7 points above 0 and 5 points below
        n0_top = 7
        n0_bottom = 5
    else:
        raise ValueError('Incorrect wavelet family. only sym6 is available')

    # Process the scaling function as the largest wavelet
    if ii_lvl == 0:
        ii_lvl = 1

    # sizes increase with ii_lvl decreasing
    n0_top *= (2 ** (ii_lvl - 1))  # -> dilation of the first top size
    n0_bottom *= (2 ** (ii_lvl - 1))  # -> sum of the "filter at each step + dilation"
    # N_wavelet = N0_top + N0_bottom
    # print('N0_top-N0_bottom = ', N0_top-N0_bottom)

    return n0_top, n0_bottom

##
# @package wavelet_propa_size
# @brief Calculate the size of the wavelet after propagation
# @author Remi Douvenot
# @date 20/07/2021
# @version V1
# @warning The limited window induces a low error (typically -70 dB) Try to improve this (increase the angle ??)
# @param[in] N_top: number of points at the top of the wavelet
# @param[in] N_bottom: number of points at the bottom of the wavelet (N_top + N_bottom = size of the wavelet)
# @param[in] x_step: longitudinal (x) step for propagation calculation
# @param[in] z: vertical (z) step for propagation calculation
# @param[in] ll: max level of decomposition
# @param[out] N_u_top: number of points at the top of the wavelet after propagation
# @param[out] N_u_bottom: number of points at the bottom of the wavelet after propagation
# @details Calculate the support of the wavelet functions to propagate. The objectives are :\n
# - add points to stay in the 90°-cone of validity of WAPE
# - have a multiple of 2^L points for the "bottom part" of the wavelet
# - have a multiple of 2^L points for the "top part" of the wavelet (top part = bottom part)
##


def wavelet_propa_size(n_bottom, x_step, z_step, ll):

    # number of points to add before propagation (at the top AND at the bottom)
    n_add = int(np.ceil(x_step * np.sin(np.pi / 4) / z_step))
    # make N_u_bottom a multiple of 2^L
    n_u_bottom = n_bottom + n_add
    remaining = n_u_bottom % (2 ** ll)
    if remaining:  # if not zero
        n_u_bottom += (2 ** ll) - remaining

    # make N_u_top a multiple of 2^L
    '''N_u_top = N_top + N_add
    remaining = (N_u_top) % (2 ** ll)
    if remaining: # if not zero
        N_u_top += (2 ** ll) - remaining'''
    # you force this way to have an odd number of wavelet coefficients: make convolutions easier to deal with
    # N_u_top = int(N_u_bottom + 2 ** ll)
    # N_u_bottom = N_u_top
    n_u_top = int(n_u_bottom)
    # take margins @todo Choose this margin wrt. desired accuracy
    n_u_bottom *= 1
    n_u_top *= 1
    # print('N_u_top-N_u_bottom = ', N_u_top-N_u_bottom)
    return n_u_top, n_u_bottom

##
# @package shift_field
# @brief Shift the field to deal with non-matching multilevel meshes
# @author Remi Douvenot
# @date 08/06/2021
# @version V1
# @param[in] u_wavelet: field corresponding to the wavelet
# @param[in] ii_lvl: current wavelet level
# @param[in] ll: max level of decomposition
# @param[in] q_y: wavelet shift along y
# @param[in] q_z: wavelet shift along y
# @param[out] u_wavelet_t: shifted field
##


def shift_field(u_wavelet, ii_lvl, ll, q_z):

    n_z = u_wavelet.size

    # space shift for a shift of q in wavelet domain q in [0,2^(L+1-l)[ ([0,2^(l-1)[ in Python)
    shift_z = q_z * (2 ** (ll + 1 - ii_lvl))
    # print('level = ', ii_lvl, 'q_y', q_y, 'q_z', q_z, 'shift_y = ', shift_y, 'shift_z = ', shift_z)
    # shift the centered wavelet along y
    u_wavelet_t = np.zeros_like(u_wavelet)
    u_wavelet_t[shift_z:n_z] = u_wavelet[0:n_z-shift_z]

    return u_wavelet_t


##
# @brief function to plot the a 2D wavelet. Useful for debug
##


def plot_2D_wavelet(wavelet_2D):
    Nx, Ny = wavelet_2D.shape

    fig = plt.figure(figsize=(13, 7))
    xx, yy = np.meshgrid(np.arange(0,Nx), np.arange(0,Ny))
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(xx, yy, wavelet_2D, rstride=1, cstride=1, cmap='coolwarm', edgecolor='none')
    ax.set_xlabel('y')
    ax.set_ylabel('z')
    ax.set_zlabel('PDF')
    ax.set_title('Diag')
    fig.colorbar(surf, shrink=0.5, aspect=5)  # add color bar indicating the PDF
    # ax.view_init(60, 35)
    plt.show()

    return 0


##
# @package init_dictionary
# @brief Initialisation of an empty dictionary of propagators
# @author Thomas Bonnafont
# @date 08/06/2021
# @version V1
# @param[in] ll: max level of decomposition
# @param[out] dictionary: empty dictionary of propagators
# @details The dictionary of propagators has a very specific shape: 1 scaling function and L wavelets with 3 orientations
# for each (start space). For each of them, 1 scaling function and L wavelets with 3 orientations for each.
##

def init_dictionary(ll):

    # number of translations at each level
    q_max_list = q_max_calculation(ll)

    # an empty list at each level
    dictionary = [[] for _ in np.arange(ll+1)]
    # fill each list with the correct number of propagators
    for ii_lvl in np.arange(0, ll+1):
        # number of translations needed in 1D direction at each level
        q_max = q_max_list[ii_lvl]
        # compute the total number of propagators needed in 2D
        # number of translations at each level
        dictionary[ii_lvl] = [[] for _ in np.arange(q_max)]

    # empty dictionary with the correct number of lists and sub-lists and sub-sub-lists and so on...
    return dictionary

