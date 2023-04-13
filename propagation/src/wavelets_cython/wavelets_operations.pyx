# This file is part of SSW-2D.
# SSW-2D is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# SSW-2D is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
# of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Foobar. If not, see
# <https://www.gnu.org/licenses/>.

# cython: infer_types=True

import numpy as np

cdef extern from "complex.h":  # import complex number library
    pass

##
# @file wavelet_operations.pyx
#
# @package calculate_dilation
# @brief Calculation of the number of propagators at each level
# @author Rémi Douvenot
# @date 14/03/2023
# @version V1
# @param[in] ll Max level of decomposition
# @param[out] q_max_list Number of propagators at each level
# @details Calculation of the dilation at each level (also the number of propagators at each level)
#
# Compilation:
# >> cython -3 -a wavelets_operations.pyx
# >> gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -ffast-math -I/usr/include/python3.10 -o wavelets_operations.so wavelets_operations.c
##


def calculate_dilation(int ll):
    dilation_list = np.zeros(ll + 1, dtype=np.intc)  # L wavelets + 1 scaling function
    # cython counterpart
    cdef int[:] dilation_list_cy = dilation_list
    # define counter
    cdef Py_ssize_t ii_lvl
    cdef int level
    cdef int tmp
    # list
    for ii_lvl in range(ll + 1):
        level = ii_lvl
        if ii_lvl == 0:  # level 0 (scaling function) corresponds to level 1 in equation below
            level = + 1
        tmp = 2 ** (level - 1)
        dilation_list_cy[ii_lvl] = tmp

    return dilation_list


##
# @file wavelet_operations.pyx
#
# @package normalized_indices
# @brief Calculation of the number of propagators at each level
# @author Rémi Douvenot
# @date 14/03/2023
# @version V1
# @param[in] ll Max level of decomposition
# @param[out] q_max_list Number of propagators at each level
# @details Give the normalized indices of each wavelet level (start and stop). Multiply by the size of the scaling
# function to obtain the actual indices.
##


def normalized_indices(int ll):
    norm_indices = np.zeros(ll + 2, dtype=np.intc)  # L wavelets + 1 scaling function
    # cython counterpart
    cdef int[:] norm_indices_cy = norm_indices
    # define counter
    cdef Py_ssize_t ii_lvl
    # list
    for ii_lvl in range(1, ll + 2):
        norm_indices_cy[ii_lvl] = 2 ** (ii_lvl - 1)

    return norm_indices
