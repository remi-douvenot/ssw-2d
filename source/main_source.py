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
# @mainpage         main_source: source generation (initial field) for SSW-2D
# @author           Rémi Douvenot, ENAC
# @date             17/06/22
# @version          0.1
#
# @section intro    introduction
#                   This document describes the code for the source generation of the SSW-2D code.
#
# @section prereq   Prerequisites.
#                   Python packages: numpy, scipy, pywavelets, matplolib
#
# @section install  Installation procedure
#                   No installation. Just run the main program.
#
# @section run      Run main_source for SSW-2D
#                   Fill the desired options in the inputs/configuration.csv file
#                   Just run the main_source via python3
#
##

##
# @package main_source
# @author R. Douvenot
# @date 07/04/2021 (created)
# @version 1.0
#
# @brief Computes the initial field for the 3D SSW software
# @warning only complex source point for now
#
# @param[in]
# Inputs are defined in the files in the "inputs" directory
# - configuration.csv that contains \n
# -- frequency  frequency, in MHz
# -- type:      source type (available: CSP)
# -- N_z:       number of vertical points
# -- z_step     vertical step
# -- x_s        horizontal position opf the source (negative)
# -- P_Tx       input power (W)
# -- G_Tx       antenna maximal gain (dBi)
# -- z_s        altitude of the source (in m)
# -- W0         width of the aperture (for CSP source type)
# @param[out] u_field (N_y,N_z)-array. Contains the initial field for 3D-SSW
##

import numpy as np
import csv
import scipy.constants as cst
import matplotlib.pyplot as plt
# import sys
from src.complex_source_point import complex_source_point
from src.plane_wave_source import plane_wave_source

# contains the source type
file_source = 'inputs/configuration.csv'


# ------------------------ #
# --- Defining classes --- #
# ------------------------ #
class ConfigSource:
    def __init__(self):
        self.n_z = 0
        self.z_step = 0
        # position of the CSP along x (real <0)
        self.x_s = 0
        # position of the CSP along z (real)
        self.z_s = 0
        # width of the complex source point (real)
        self.W0 = 0
        # radiated power in W (real)
        self.P_Tx = 0
        # max gain in dBi (real)
        self.G_Tx = 0
        # wavenumber in m
        self.k0 = 0
        # altitude of the source in m
        self.z_s = 0
        # width of the aperture W0 (for CSP)
        self.W0 = 0


# ---------- END --------- #
# --- Defining classes --- #
# ------------------------ #


# --------------------------- #
# --- Reading source type --- #
# --------------------------- #
f_source = open(file_source, newline='')
source_tmp = csv.reader(f_source)
for row in source_tmp:
    if row[0] == 'N_z':
        ConfigSource.n_z = np.int32(row[1])
    elif row[0] == 'z_step':
        ConfigSource.z_step = np.float64(row[1])
    elif row[0] == 'x_s':  # position along x --> must be <0
        ConfigSource.x_s = np.float64(row[1])
    elif row[0] == 'type':
        source_type = row[1]
    elif row[0] == 'frequency':
        freq = np.float64(row[1])*1e6
    elif row[0] == 'P_Tx':  # radiated power
        ConfigSource.P_Tx = np.float64(row[1])
    elif row[0] == 'G_Tx':  # antenna max gain (dBi)
        ConfigSource.G_Tx = np.float64(row[1])
    elif row[0] == 'W0':  # waist of the CSP
        ConfigSource.W0 = np.float64(row[1])
    elif row[0] == 'z_s':  # position along z
        ConfigSource.z_s = np.float64(row[1])
    elif row[0] == 'Property':  # first line
        pass
    else:
        raise ValueError(['Input file of the geometry is not valid. Input "' + row[0]+'" not valid'])

if ConfigSource.x_s >= 0:
    raise ValueError(['Source must be in x<0 half-space (x_s < 0)'])
# ------------ END ---------- #
# --- Reading source type --- #
# --------------------------- #


# ---------------------------- #
# --- CREATE INITIAL FIELD --- #
# ---------------------------- #

# wavenumber
ConfigSource.k0 = 2 * cst.pi * freq / cst.c
# compute E field
e_field = complex_source_point(ConfigSource)
#e_field =plane_wave_source(ConfigSource)


# ------------ END ----------- #
# --- CREATE INITIAL FIELD --- #
# ---------------------------- #


# -------------------------- #
# --- Saving the results --- #
# -------------------------- #

# saving configuration
rows = [['frequency', str(freq*1e-6), 'MHz'],
        ['N_z', str(ConfigSource.n_z)],
        ['z_step', str(ConfigSource.z_step), 'm'],
        ['x_s', str(ConfigSource.x_s), 'm'],
        ['z_s', str(ConfigSource.z_s), 'm']]

with open('./outputs/configuration.csv', 'w', newline='') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)
    write.writerows(rows)

# saving the field
np.savetxt('./outputs/E_field.csv', e_field, delimiter=',')
np.save('./outputs/E_field.npy', e_field)

# ---------- END ----------- #
# --- Saving the results --- #
# -------------------------- #

# --- plot --- #

'''plt.figure()
ax = plt.subplot(111)
e_field_db = 20 * np.log10(np.abs(e_field))
v_max = np.max(e_field_db)
v_min = v_max - 100
print('Max field = ', np.round(v_max, 2), 'dBV/m')
z_vect = np.linspace(0, Geom.z_step*Geom.n_z, num=Geom.n_z, endpoint=False)
plt.plot(e_field_db, z_vect)
plt.xlim(v_min, v_max)
plt.ylim(0, Geom.z_step*Geom.n_z)
plt.xlabel('E field (dBV/m)', fontsize=14)
plt.ylabel('Altitude (m)', fontsize=14)
plt.title('Initial field')
plt.show()'''
