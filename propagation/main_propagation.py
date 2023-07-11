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
# @mainpage         main_propagation: Core program of the split-step wavelet SSW-2D package
# @author           Rémi Douvenot, ENAC
# @date             2022/06/17 -- last modification 2022/11/10
# @version          0.2
#
# @section intro    Introduction
#                   This document describes the code for the propagation by the SSW-2D code.
#                   This code calculates the electric field propagated on long distances using the split-step wavelet or
#                   the wavelet-to-wavelet method.
#
# @section prereq   Prerequisites.
#                   Python packages: numpy, scipy, pywavelets, matplolib
#
# @section install  Installation procedure
#                   No installation. Just run the main program.
#
# @section run      Run main_propagation for SSW-2D \n
#                   1/ Fill the desired options in the inputs/configuration.csv file \n
#                   2/ Run main_terrain and main_source with the desired options \n
#                   3/ Just run the main_terrain via python3. The final field and the wavelet decomposition of all the
#                   2D field are stored in the outputs directory. \n
#
#                   OR, alternatively \n
#                   1/ run the GUI module. This package is launched when pressing "run simulation"
#
##

##
# @file main_propagation.py
#
# @package: main_propagation
# @author: R. Douvenot
# @date: 19/07/2021
# @version: work in progress
#
# @brief Propagates a field with the 2D SSW or 2D WWP method
# @details Propagates a field with the 2D SSW or WWP method. Inputs are first read from input files and SSW_3D package
# is launched.
#
# @param[in]
# Inputs are defined in the files in the "inputs" directory
# - configuration.csv that contains \n
# -- method:        chosen propagation method. Can be "SSW" or "WWP"
# -- N_x:           number of propagation steps \n
# -- N_z:           number of vertical points \n
# -- x_step:        step along the main propagation direction \n
# -- z_step:        step along the vertical in m \n
# -- frequency:     frequency in MHz
# -- Max compression error: Maximum admissible error due to wavelet compression at the last vertical step, in dB
# -- wavelet level: maximum level of the wavelet multilevel decomposition
# -- wavelet family:    wavelet family
# -- apodisation % in z: part of the window dedicated to apodisation along the vertical (portion of 1)
# -- image layer % in z: part of the window dedicated to image layer (if there is a ground) (portion of 1)
# -- ground:        ground type: PEC, dielectric or None
# -- epsr           ground relative permittivity (if ground = dielectric)
# -- sigma          ground conductivity in S (if ground = dielectric)
# Initial field is read in ../source/outputs directory (output of main_source.py function)
#
# @param[out]       the outputs are saved in the /output/ directory.
#                   - E_field.npy contains the electric field on the last vertical of calculation.
#                   - wv_total.npy contains the total electric field in the shape of wavelet coefficients.
#
# @warning: only atmosphere is accounted with WWP. No ground or relief.
##


import numpy as np
import time
import scipy.constants as cst
from src.wavelets.wavelet_operations import compute_thresholds
from src.propagation.ssw_2d import ssw_2d_light
from src.propagation.wwp_2d import wwp_2d_light
import shutil  # to make file copies
# where config is defined
from src.classes_and_files.read_files import read_config, read_source, read_relief
from src.atmosphere.genere_n_profile import generate_n_profile
import pywt
import matplotlib.pyplot as plt

# -------------------------------------------------- #
# --- Declare the files where inputs are written --- #
# ---------------- DO NOT MODIFY ------------------- #

# input: main configuration file
file_configuration = './inputs/configuration.csv'
# input: the source information
file_source_config = '../source/outputs/configuration.csv'
# input: initial electric field
file_E_init = '../source/outputs/E_field.csv'
# input: refractivity phase screens
file_refractivity = '../refractivity/outputs/phase_screens.csv'
# input: relief configuration
file_relief_config = '../terrain/inputs/conf_terrain.csv'
# input: relief vector
file_relief = '../terrain/outputs/z_relief.csv'

# copy the configuration
file_output_config = './outputs/configuration.csv'
shutil.copyfile(file_configuration, file_output_config)

# -------------------- END ------------------------- #
# --- Declare the files where inputs are written --- #
# ---------------- DO NOT MODIFY ------------------- #

# --- Define and fill the config variable that contains all the simulation parameters --- #
config = read_config(file_configuration)
# --------------------------------------------------------------------------------------- #

# --- Read initial field --- #
e_field, config.z_s = read_source(config, file_source_config, file_E_init)
# -------------------------- #

# --- Calculate u_0 from E_init (normalised in infinity norm to have max(|u_0|) = 1) --- #
k0 = 2*np.pi*config.freq/cst.c
u_0 = e_field * np.sqrt(k0*(-config.x_s)) * np.exp(1j * k0 * (-config.x_s))
u_infty = np.max(np.abs(u_0))  # norm infinity of the initial field
u_0 /= u_infty  # put max at 1 to avoid numerical errors
# -------------------------------------------------------------------------------------- #

'''# Wavelet test
u_0 *= 0
UU_0 = pywt.wavedec(u_0, config.wv_family, 'per', config.wv_L)
# --- 2 wavelets field --- #
UU_0[0][10] = 1.0
u_0 = pywt.waverec(UU_0, config.wv_family, 'per')'''

# ----------------------------------------- #
# --- Creating refraction phase screens --- #
# ----------------------------------------- #
# n_refraction = np.ones(config.N_z)
n_refraction = generate_n_profile(config)

# -------------- END ---------------------- #
# --- Creating refraction phase screens --- #
# ----------------------------------------- #

# --- Computing wavelet thresholds --- #
# following Bonnafont et al. IEEE TAP, 2021, eqs .(35) and (36)
# calculation of the compression thresholds. Inf norm of u_0 is not necessary (=1 because of normalisation)
config.V_s, config.V_p = compute_thresholds(config.N_x, config.max_compression_err)  # threshold on signal and library
# ------------------------------------ #

# ---------------------- #
# --- 2D Propagation --- #
# ---------------------- #

# SSW
t_propa_s = time.process_time()
u_final = ssw_2d_light(u_0, config)
t_propa_f = time.process_time()
print('Total '+config.method+' (ms)', np.round((t_propa_f-t_propa_s)*1e3))

# test C++ code
# TODO Jeremy : coder en C le code SSW 2D simplifie.
t_propa_s = time.process_time()
# u_final_cpp = ssw_2d_cpp(u_0, config, n_refraction, ii_vect_relief)
t_propa_f = time.process_time()
print('Total '+config.method+' (ms)', np.round((t_propa_f-t_propa_s)*1e3))


# WWP
t_propa_s = time.process_time()
u_final = wwp_2d_light(u_0, config)
t_propa_f = time.process_time()
print('Total '+config.method+' (ms)', np.round((t_propa_f-t_propa_s)*1e3))

# test C++ code
# TODO Jeremy : coder en C le code SSW 2D simplifie.
t_propa_s = time.process_time()
# u_final_cpp = wwp_2d_cpp(u_0, config, n_refraction, ii_vect_relief)
t_propa_f = time.process_time()
print('Total '+config.method+' (ms)', np.round((t_propa_f-t_propa_s)*1e3))


# TODO : comparer les u_final

# TODO : comparer les u_final en dB eventuellement (precision dB fixee par la compression)
with np.errstate(divide='ignore'):
    data_dB = 20*np.log10(np.abs(u_field))
    v_max = data_dB.max()

# ----------- END ------------ #
# --- save the output data --- #
# ---------------------------- #

'''# SSF test
u_final = wavelet_propagation(u_0,config)
# de-normalise in infinity norm
u_final *= u_infty

# e_field2 = u_final * (k0*np.sqrt(-x_s)) * np.exp(1j * k0 * (-x_s+x_max))
e_field2 = u_final * np.exp(-1j * k0 * x_max)
data_dB = 20*np.log10(np.abs(e_field2))
v_max = data_dB.max()
print('max field = ', v_max)

u_infty = np.max(np.abs(e_field2))
E_diff = (e_field - e_field2)/u_infty

np.save('./outputs/e_field', e_field2)'''

