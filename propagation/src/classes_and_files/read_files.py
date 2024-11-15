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
# @package read_config
# @author Remi Douvenot
# @date 19/07/2021
# @brief Fill the config class with the input from input file
##

# where config class is defined
import csv
import numpy as np
import scipy.constants as cst
import datetime

from src.classes_and_files.classes import *


def read_config(file_configuration):
    # ----------------------------- #
    # --- Reading configuration --- #
    # ----------------------------- #
    with open(file_configuration, newline='') as f_config:
        file_tmp = csv.reader(f_config)
        for row in file_tmp:
            if row[0] == 'method':  # SSF or SSW or WWP or WWP-H
                config.method = row[1]
            elif row[0] == 'N_z':
                config.N_z = int(row[1])
            elif row[0] == 'N_x':
                config.N_x = int(row[1])
            elif row[0] == 'x_step':
                config.x_step = float(row[1])
            elif row[0] == 'z_step':
                config.z_step = float(row[1])
            elif row[0] == 'frequency':
                config.freq = float(row[1]) * 1e6  # freq in MHz
            elif row[0] == 'polarisation':
                config.polar = row[1]  # 'TE' or 'TM'
            elif row[0] == 'Max compression error':
                config.max_compression_err = float(row[1])  # Max compression error
            elif row[0] == 'wavelet level':
                config.wv_L = int(row[1])  # Max compression error
            elif row[0] == 'wavelet family':
                config.wv_family = row[1]  # Max compression error
            elif row[0] == 'apodisation window':
                config.apo_window = row[1]  # Type of the apodisation window
            elif row[0] == 'apodisation size':
                config.apo_z = float(row[1])  # apodisation size along z
            elif row[0] == 'image size':
                config.image_layer = float(row[1])  # image layer size (in ground) along z
            elif row[0] == 'ground':
                config.ground = row[1]  # ground type
            elif row[0] == 'epsr':
                config.epsr = float(row[1])  # ground relative permittivity
            elif row[0] == 'sigma':
                sigma = float(row[1])  # ground conductivity
            elif row[0] == 'atmosphere':
                config.atmosphere = row[1]  # atmospheric profile type
            elif row[0] == 'c0':
                config.c0 = float(row[1])  # standard atm gradient
            elif row[0] == 'delta':
                config.delta = float(row[1])  # evaporation duct height
            elif row[0] == 'zb':
                config.zb = float(row[1])  # base height of a trilinear duct
            elif row[0] == 'c2':
                config.c2 = float(row[1])  # gradient in a trilinear duct
            elif row[0] == 'zt':
                config.zt = float(row[1])  # thickness of a trilinear duct
            elif row[0] == 'turbulence':
                config.turbulence = row[1]
            elif row[0] == 'Cn2':
                config.Cn2 = float(row[1])
            elif row[0] == 'L0':
                config.L0 = float(row[1])
            elif row[0] == 'py_or_cy':
                config.py_or_cy = row[1]
            elif row[0] == 'Property':
                pass  # first line
            elif row[0] == 'dynamic':  # only used for HMI plots
                pass  # first line
            elif row[0] == 'atmosphere_datetime': # datetime at which the atmospherical data should be taken (UTC)
                config.atmosphere_datetime = datetime.datetime.fromisoformat(row[1])
            elif row[0] == 'P':
                # Convert "43.76654;65.875675" to (43.76654, 65.875675) tuple of floats
                config.P = tuple([float(l) for l in row[1].split(';')])
            elif row[0] == 'Q':
                # Same
                config.Q = tuple([float(l) for l in row[1].split(';')])
            else:
                raise ValueError(['Input file of the configuration is not valid. Input "' + row[0] + '" not valid'])
    # check for some values
    if config.apo_z < 0 or config.image_layer < 0 or config.apo_z > 0.5 or config.image_layer > 0.5:
        raise ValueError(['Apodisation and image layer must be in [0,0.5] (percentage of the total field size'])

    if (config.ground != 'NoGround') & (config.ground != 'PEC') & (config.ground != 'Dielectric'):
        raise ValueError(['Ground must be chosen among: NoGround, PEC, or Dielectric'])

    if (config.method != 'SSW') & (config.method != 'WWP') & (config.method != 'WWP-H') & (config.method != 'SSF'):
        raise ValueError(['Method must be chosen among: SSW or WWP or WWP-H or SSF'])
    # ------------ END ------------ #
    # --- Reading configuration --- #
    # ----------------------------- #

    # check apodisation window
    if config.apo_window == 'Hanning':
        print('Hanning apodisation window')
    else:
        raise (ValueError([config.apo_window, 'is not a valid apodisation type']))

    # --- Check the size of the vectors, multiple of 2^n --- #
    n_scaling_fct = 2 ** config.wv_L
    modulo_nz = config.N_z % n_scaling_fct
    if modulo_nz != 0:
        raise (ValueError(['N_z must be multiple of', n_scaling_fct, ' = 2^L']))

    # epsr_effective is used
    config.epsr += -1j * sigma / (2*cst.pi*config.freq*cst.epsilon_0)

    # Check for required value for real atmospheres
    if config.atmosphere == 'era5' and (config.P == None or config.Q == None or config.atmosphere_datetime == None):
        raise ValueError(f'When using {config.atmosphere} atmosphere, P, Q and atmosphere_datetime should be set')

    return config


def read_source(config, file_source_config, file_e_init):
    # ------------------------------ #
    # --- Reading source E-field --- #
    # ------------------------------ #

    # --- Configuration first --- #
    with open(file_source_config, newline='') as f_source_config:
        file_tmp = csv.reader(f_source_config)
        for row in file_tmp:
            # geometry must match with the source generation
            if row[0] == 'N_z':
                n_z = int(row[1])
                if n_z != config.N_z:
                    raise ValueError(['n_z value does not match with source generation'])
            elif row[0] == 'z_step':
                z_step = float(row[1])
                if z_step != config.z_step:
                    raise ValueError(['z_step value does not match with source generation'])
            # geometry must match with the source generation
            elif row[0] == 'frequency':
                freq = float(row[1]) * 1e6  # freq in MHz
                if np.abs(freq - config.freq) > 2*2.22e-16:
                    raise ValueError(['frequency ', freq, ' MHz value does not match with source generation',
                                    config.freq, ' MHz'])
            # x_s is the source position in the x direction (must be <0)
            elif row[0] == 'x_s':
                config.x_s = float(row[1])  # distance in m
                if config.x_s >= 0:
                    raise ValueError(['Source position along x must be <0'])
            # z_s is the source altitude in the z direction
            elif row[0] == 'z_s':
                z_s = float(row[1])  # altitude in m
            # property = first line to skip
            elif row[0] == 'Property':
                pass  # first line
            else:
                raise ValueError(['Output file of the source generation is not valid. Input "' + row[0] + '" not valid'])

    # --- The electric field itself --- #
    e_field = np.loadtxt(file_e_init, delimiter=',', dtype="complex")
    n_z = e_field.size
    # size of the electric field must match the geometry parameters
    if n_z != config.N_z:
        raise ValueError(['N_z value does not match with saved initial field'])

    # ------------- END ------------ #
    # --- Reading source E-field --- #
    # ------------------------------ #

    return e_field, z_s


def read_relief(config, file_relief_config, file_relief):

    # ---------------------- #
    # --- Reading relief --- #
    # ---------------------- #

    # --- Configuration first --- #
    with open(file_relief_config, newline='') as f_source_config:
        file_tmp = csv.reader(f_source_config)
        for row in file_tmp:
            # geometry must match with the relief generation
            if row[0] == 'N_x':
                n_x = int(row[1])
                if n_x != config.N_x:
                    raise ValueError(['n_x value does not match with relief'])
            # geometry must match with the source generation
            elif row[0] == 'x_step':
                x_step = float(row[1])  # horizontal step in m
            elif row[0] == 'z_max_relief':
                z_max_relief = float(row[1])  # max relief in m
                if z_max_relief > config.z_step*config.N_z:
                    raise ValueError(['Relief is higher than the computation domain!'])
            elif row[0] == 'type':
                config.type = row[1]  #
            elif row[0] == 'iterations':
                config.iterations = int(row[1])  #
            elif row[0] == 'center':
                config.center = float(row[1])  #
            elif row[0] == 'width':
                config.width = float(row[1])  #
            elif row[0] == 'P':
                # Convert "43.76654;65.875675" to (43.76654, 65.875675) tuple of floats
                config.P = tuple([float(l) for l in row[1].split(';')])
            elif row[0] == 'Q':
                # Same
                config.Q = tuple([float(l) for l in row[1].split(';')])
            elif row[0] == 'Property':
                pass  # first line
            else:
                raise ValueError(['Output file of the relief generation is not valid. Input "' + row[0] + '" not valid'])

    # --- The relief itself --- #
    z_relief = np.loadtxt(file_relief, delimiter=',', dtype="float")
    # size of the electric field must match the geometry parameters
    z_relief = config.z_step*np.round(z_relief/config.z_step)

    return z_relief
