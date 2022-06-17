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
# @file compute_thresholds.py
#
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
#
##


def compute_thresholds(n_x, max_error_db):
    max_error = 10 ** (max_error_db / 20)
    threshold_v_s = max_error/(2 * n_x)
    threshold_v_m = max_error/(2 * n_x)
    return threshold_v_s, threshold_v_m