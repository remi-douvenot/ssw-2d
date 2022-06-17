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
# @file thresholding.py
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
