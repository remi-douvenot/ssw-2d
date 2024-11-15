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
# @mainpage         main_terrain: terrain (relief) generation for SSW-2D
# @author           Rémi Douvenot, ENAC
# @date             17/06/22
# @version          0.1
#
# @section intro    introduction
#                   This document describes the code for the terrain generation of the SSW-2D code.
#
# @section prereq   Prerequisites.
#                   Python packages: numpy, scipy
#
# @section install  Installation procedure
#                   No installation. Just run the main program.
#
# @section run      Run main_source for SSW-2D
#                   Fill the desired options in the inputs/configuration.csv file
#                   Just run the main_terrain via python3
#
##

##
# @package main_terrain
# @author R. Douvenot
# @date 11/01/2021 (created)
# @version 1.0
#
# @brief Computes the terrain for the 2D SSW software
# source: https://arpit.substack.com/p/1d-procedural-terrain-generation
#
# @param[in]
# Inputs are defined in the files in the "inputs" directory
# - terrain.csv that contains \n
# -- N_x:       number of horizontal points \n
# -- z_step:    step along the vertical in m \n
# -- z_max_relief: max altitude of the relief in m \n
# -- iterations: number of scales in the multiscale procedural generation \n
# -- width:     number of pts defining the width of the relief (Triangle) in pts \n
# -- center:    number of pts along x at which is the center of the relief (Triangle) in pts \n
#
# @param[out] z_relief (N_x)-array. Contains altitudes of the relief. Scaled between 0 and z_max_relief
##


import numpy as np
import scipy.constants as cst
import matplotlib.pyplot as plt
import shutil  # to make file copies

from src.terrain_gen import superposed
from src.read_config import read_config
import src.relief as rel

# contains the source type
file_terrain = './inputs/conf_terrain.csv'
# read the inputs
config = read_config(file_terrain)
# no relief = Plane relief
if config.type == 'Plane':
    # no relief
    z_relief = np.zeros(config.N_x+1)  # N_x + 1 from 0 to N_x (included)

# multiscale random relief
elif config.type == 'Superposed':
    # create the multiscale relief by superposition
    z_relief = np.array(superposed(config))
    # scale the relief between 0 and max relief
    z_relief -= z_relief.min()
    z_relief *= config.z_max_relief / z_relief.max()

# triangle relief
elif config.type == 'Triangle':
    # scale the relief between 0 and max relief
    z_relief = np.zeros(config.N_x+1)
    triangle_start = int((config.center-config.width/2) / config.x_step)
    triangle_end = int((config.center+config.width/2) / config.x_step)
    triangle_top = int(config.center / config.x_step)
    x_tri = [0, triangle_start, triangle_top, triangle_end, config.N_x]
    z_tri = [0, 0, config.z_max_relief, 0, 0]
    z_relief = np.interp(np.arange(0, config.N_x+1), x_tri, z_tri)
# IGN
elif config.type == 'IGN':
    # Retrieve the height dataset from IGN
    ds = rel.get_ign_height_profile(config.P, config.Q, config.N_x+1)
    #  Convert it to a np.array
    z_relief = np.array(ds.height)
# Bing
elif config.type == 'Bing':
    # Retrieve the height dataset from Bing
    ds = rel.get_bing_height_profile(config.P, config.Q, config.N_x+1)
    #  Convert it to a np.array
    z_relief = np.array(ds.height)
else:
    z_relief = np.zeros(config.N_x+1)
    raise (ValueError(['terrain type not coded']))

# -------------------------- #
# --- Saving the results --- #
# -------------------------- #

# saving the terrain
np.savetxt('./outputs/z_relief.csv', z_relief, delimiter=',')

# ---------- END ----------- #
# --- Saving the results --- #
# -------------------------- #

#
# plt.figure()
# ax = plt.subplot(111)
# x_relief = np.arange(config.N_x+1)
# plt.plot(x_relief, z_relief)
# ax.fill_between(x_relief, z_relief, where=z_relief > 0, facecolor='black')
#
# plt.show()


