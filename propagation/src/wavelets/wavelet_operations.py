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
# @file wavelet_operations.py
#
# @package thresholding
# @author T. Bonnafont
# @brief This function applies a hard threshold on a wavelet decomposition. Necessitates changes tuple -> array -> tuple
# @version 1.0
# @param[in] coeffs
# @param[in] threshold
# @param[out] coeffs_t
##

import pywt
import numpy as np
from scipy.sparse import coo_matrix, dok_array  # for sparsify


# from scipy.sparse import dok, dok_matrix


def thresholding(coeffs, threshold):
    # Obtain an array of the wavelet coefficient
    coeffs_arr, coeffs_slice = pywt.coeffs_to_array(coeffs)

    # Apply the threshold on the array
    coeffs_arr = pywt.threshold(coeffs_arr, threshold, mode='hard')
    # coeffs_arr_dok = dok_matrix(coeffs_arr)
    # Change the array to a wavelet coeffs list of array form
    coeffs_t = pywt.array_to_coeffs(coeffs_arr, coeffs_slice, output_format='wavedec')

    return coeffs_t


##
# @package q_max_calculation
# @brief Calculation of the number of propagators at each level
# @author Thomas Bonnafont
# @date 08/06/2021
# @version V1
# @param[in] ll Max level of decomposition
# @param[out] q_max_list Number of propagators at each level
# @details Calculation of the number of propagators at each level with respect to the largest decomposition level
##


def q_max_calculation(ll):
    q_max_list = np.zeros(ll + 1, dtype=int)  # L wavelets + 1 scaling function
    for ii_lvl in np.arange(0, ll + 1):
        level = ii_lvl
        if ii_lvl == 0:  # level 0 (scaling function) corresponds to level 1 in equation below
            level = + 1
        q_max_list[ii_lvl] = (2 ** (level - 1))

    return q_max_list


##
# @package: sparsify
# @author: R. Douvenot
# @date: 09/06/2021
# @version: V1.0
#
# @brief Put a wavelet decomposition in a sparse shape (coo format)
# @param[in] wv_x Wavelet decomposition
# @param[out] wv_x_sparse Sparse wavelet decomposition
##


def sparsify(wv_x):
    # max level of decomposition
    ll = len(wv_x) - 1
    # creation of the empty list
    wv_x_sparse = [[]] * (ll + 1)
    # fill the scaling function
    # fill the wavelet levels
    for ii_lvl in np.arange(0, ll + 1):
        wv_x_sparse[ii_lvl] = coo_matrix(wv_x[ii_lvl])
    return wv_x_sparse


##
# @package: sparsify_dok
# @author: R. Douvenot
# @date: 24/03/2023
# @version: V1.0
#
# @brief Put a wavelet decomposition in a sparse shape (dok format)
# @param[in] wv_x Wavelet decomposition
# @param[out] wv_x_sparse Sparse wavelet decomposition
##


def sparsify_dok(wv_x):
    # max level of decomposition
    ll = len(wv_x) - 1
    # creation of the empty list
    wv_x_sparse = [[]] * (ll + 1)
    # fill the scaling function
    # fill the wavelet levels
    for ii_lvl in np.arange(0, ll + 1):
        uu = np.asmatrix(wv_x[ii_lvl])
        wv_x_sparse[ii_lvl] = dok_array(uu)
    return wv_x_sparse


##
# @package: unsparsify_dok
# @author: R. Douvenot
# @date: 09/06/2021
# @version: V1.0
#
# @brief Put a wavelet decomposition in a sparse shape (coo format)
# @param[in] wv_x Wavelet decomposition
# @param[out] wv_x_sparse Sparse wavelet decomposition
##


def unsparsify_dok(wv_x_sparse):
    # max level of decomposition
    ll = len(wv_x_sparse) - 1
    # creation of the empty list
    wv_x = [[]] * (ll + 1)
    # fill the scaling function
    # fill the wavelet levels
    for ii_lvl in np.arange(0, ll + 1):
        wv_x[ii_lvl] = np.squeeze(np.asarray(wv_x_sparse[ii_lvl].todense()))
    return wv_x


##
# @package compute_threshold
# @author T. Bonnafont
# @author Remi Douvenot
# @date 29/03/18 creation. Last modif 03/05/2021
# @version 1.0
# @brief Compute the threshold values for wavelet compressions
# @details Compute the threshold values for wavelet compressions V_s (for the signal) and V_p (for the propagator)
# from the number of iterations and the maximal admissible error
#
# @param[in] N_x Number of iterations in the propagation process
# @param[in] error_ex_dB Maximum acceptable error from wavelet compression (in dB)
#
# @param[out] threshold_V_s Compression threshold we apply on the signal
# @param[out] threshold_V_p Compression threshold on the propagator
##


def compute_thresholds(n_x, max_error_db):
    max_error = 10 ** (max_error_db / 20)
    threshold_v_s = max_error / (2 * n_x)
    threshold_v_m = max_error / (2 * n_x)
    return threshold_v_s, threshold_v_m


##
# @brief function that removes the first wavelet coefficients that correspond to the image layer
# @author R. Douvenot
# @package remove_image_coef
# @date 13/12/22
# @version OK
#
# @details Function that recompose the total wavelet decomposition from ssw and wwp parts.
# def remove_image_coef(w_x, w_ssw, w_wwp, config):
#
# @params[in,out] w_x : current wavelet decomposition (to update)
# @params[in] config : configuration of the simulation
##


def remove_image_coef(w_x, config):
    ll = config.wv_L
    nn = config.N_z
    n_im = config.N_im
    # dilation at each level
    q_max = q_max_calculation(ll)
    dilation = 2 ** ll / q_max
    # on each level (including scaling), fill appropriately
    for ii_l in np.arange(0, ll + 1):
        # take into account the dilation at level ii_l
        p_im = int(n_im / dilation[ii_l])
        # remove the useless part (size = N_im --> remove image and top layers)
        w_x[ii_l] = w_x[ii_l][p_im:]

    return w_x


##
# @brief function that hybrids SSW and WWP wavelet decompositions
# @author R. Douvenot
# @package hybrid_ssw_wwp
# @date 13/12/22
# @version OK
#
# @details Function that recompose the total wavelet decomposition from ssw and wwp parts.
# def remove_image_coef(w_x, w_ssw, w_wwp, config):
#
# @params[in,out] w_x : current wavelet decomposition (to update)
# @params[in] config : configuration of the simulation
##


def hybrid_ssw_wwp(w_ssw_x, w_wwp_x, config):
    ll = config.wv_L
    nn = config.N_z
    n_im = config.N_im
    # dilation at each level
    q_max = q_max_calculation(ll)
    dilation = 2 ** ll / q_max
    # on each level (including scaling), fill appropriately
    for ii_l in np.arange(0, ll + 1):
        # take into account the dilation at level ii_l
        p_im = int(n_im / dilation[ii_l])
        # copy the WWP coefficients in SSW
        w_ssw_x[ii_l][2 * p_im:] = w_wwp_x[ii_l][:p_im]

    return w_ssw_x


##
# @brief function that extracts the SSW part from the total wavelet decomposition is WWP-H
# @author R. Douvenot
# @package extract_ssw
# @date 14/12/22
# @version OK
#
# @details Function that extracts the SSW part from the total wavelet decomposition is WWP-H
# def extract_ssw(w_ssw_x, w_wwp_x, config):
#
# @params[in, out] w_ssw : ssw wavelet decomposition (to update)
# @params[in] w_x : current wavelet decomposition (to update)
# @params[in] config : configuration of the simulation
##


def extract_ssw(w_ssw, w_x, config):
    ll = config.wv_L
    nn = config.N_z
    n_im = config.N_im
    # dilation at each level
    q_max = q_max_calculation(ll)
    dilation = 2 ** ll / q_max
    # on each level (including scaling), fill appropriately
    for ii_l in np.arange(0, ll + 1):
        # take into account the dilation at level ii_l
        p_im = int(n_im / dilation[ii_l])
        # copy the WWP coefficients in SSW (total size = image layer + 2 * image layer size)
        w_ssw[ii_l] = w_x[ii_l][:3 * p_im]

    return w_ssw


##
# @brief function that recompose the total wavelet decomposition from ssw and wwp parts
# @author R. Douvenot
# @package assemble_ssw_wwp
# @date 13/12/22
# @version OK
#
# @details Function that recompose the total wavelet decomposition from ssw and wwp parts.
# def assemble_ssw_wwp(w_x, w_ssw, w_wwp, config):
#
# @params[in] w_x : current wavelet decomposition (to update)
# @params[in] w_ssw : wavelet decomposition the SSW field
# @params[in] w_wwp : wavelet decomposition the WWP field
# @params[in] config : configuration of the simulation
##


def assemble_ssw_wwp(w_x, w_ssw, w_wwp, config):
    ll = config.wv_L
    nn = config.N_z
    n_im = config.N_im
    # dilation at each level
    q_max = q_max_calculation(ll)
    dilation = 2 ** ll / q_max
    # on each level (including scaling), fill appropriately
    for ii_l in np.arange(0, ll + 1):
        # take into account the dilation at level ii_l
        p_im = int(n_im / dilation[ii_l])
        # 1/ fill the SSW part (layer image + SSW part of same size)
        w_x[ii_l][0: 2 * p_im] = w_ssw[ii_l][0:2 * p_im]
        # 2/ fill the WWP part (for n_im to top)
        w_x[ii_l][2 * p_im:] = w_wwp[ii_l][:]

    return w_x


##
# @brief function that extracts the SSW part from the total wavelet decomposition is WWP-H
# @author R. Douvenot
# @package extract_ssw
# @date 14/12/22
# @version OK
#
# @details Function that extracts the SSW part from the total wavelet decomposition is WWP-H
# def extract_ssw(w_ssw_x, w_wwp_x, config):
#
# @params[in, out] w_ssw : ssw wavelet decomposition (to update)
# @params[in] w_x : current wavelet decomposition (to update)
# @params[in] config : configuration of the simulation
##

def disassemble_ssw_wwp(w_x, w_ssw, w_wwp, config):
    #
    ll = config.wv_L
    nn = config.N_z
    n_im = config.N_im
    # dilation at each level
    q_max = q_max_calculation(ll)
    dilation = 2 ** ll / q_max
    # on each level (including scaling), fill appropriately
    for ii_l in np.arange(0, ll + 1):
        # take into account the dilation at level ii_l
        p_im = int(n_im / dilation[ii_l])
        # copy the WWP-H coefficients in SSW (total size = image layer + 2 * image layer size)
        w_ssw[ii_l] = w_x[ii_l][:3 * p_im]
        # copy the WWP-H coefficients in WWP (total size = N_z - 3 * image layer size)
        w_wwp[ii_l] = w_x[ii_l][2 * p_im:]

    return w_ssw, w_wwp
