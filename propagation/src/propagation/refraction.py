##
# @brief function that applies half a phase screen before or after propagation
# @author R. Douvenot
# @package apply_refractive_index
# @date 10/09/21
# @version OK
#
# @details Function that applies half a phase screen before or after propagation.
# def apply_refractive_index(u_x, n_index, config):
#
# @params[in] u_x : reduced electric field (complex array)
# @params[in] n_index : phase screen (real array)
# @params[in] config : class with the parameters
# @params[in] u_x : reduced electric field (complex array)
##


import scipy.constants as cst
import numpy as np
from src.wavelets.wavelet_operations import q_max_calculation


def apply_refractive_index(u_x, n_index, config):

    k0 = 2*cst.pi*config.freq / cst.c

    # apply the phase screen of one step delta_x
    # half the refraction applied before and after propagation

    u_x *= np.exp(-1j * k0 * (n_index-1)/2 * config.x_step)

    return u_x


##
# @brief function that applies a phase screen on the wavelet coefficients right after propagation
# @author R. Douvenot
# @package apply_refractive_index_wavelets
# @date 10/09/21
# @version OK
#
# @details Function that applies half a phase screen before or after propagation.
# def apply_refractive_index_wavelets(u_x, n_index, config):
#
# @params[in,out] w_x : wavelet decomposition of the electric field (complex array)
# @params[in] n_index : phase screen (real array)
# @params[in] config : class with the parameters
##


def apply_refractive_index_wavelet(w_x, n_index, config):

    k0 = 2*cst.pi*config.freq / cst.c
    # number of q_max per level
    q_max = q_max_calculation(config.wv_L)
    # decimation coefficient per level
    decimation = (2**config.wv_L/q_max).astype(int)
    # apply the phase screen on one step delta_x
    for ii_l in np.arange(0, config.wv_L+1):
        w_x_ll = w_x[ii_l]
        delta = decimation[ii_l]
        w_x_ll *= np.exp(-1j * k0 * (n_index[::delta]-1) * config.x_step)
        w_x[ii_l] = w_x_ll

    return w_x


##
# @brief function that applies a turbulent phase screen (defined in phase directly)
# @author V. Darchy
# @package apply_phi_turbulent
# @date 10/09/21
# @version OK
#
# @details function that applies a turbulent phase screen (defined in phase directly)
# def apply_phi_turbulent(u_x, phi_turbulent, config):
#
# @params[in,out] u_x : electric field (complex array)
# @params[in] phi_turbulent : turbulent phase screen (real array)
# @params[in] config : class with the parameters
##
def apply_phi_turbulent(u_x, phi_turbulent, config):

    # apply the turbulent phase screen of one step delta_x
    # half the refraction applied before and after propagation
    k0 = 2 * cst.pi * config.freq / cst.c
    u_x *= np.exp(-1j * phi_turbulent)

    return u_x

