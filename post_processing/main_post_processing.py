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
# @mainpage         main_post_processing: Plot the result from the split-step wavelet propagation module
# @author           Rémi Douvenot, ENAC
# @date             17/06/22
# @version          0.1
#
# @section intro    Introduction
#                   This document describes the code for the post-processing of the SSW-2D data
#
# @section prereq   Prerequisites.
#                   Python packages: numpy, scipy, pywavelets, matplolib
#
# @section install  Installation procedure
#                   No installation. Just run the main program.
#
# @section run      Run main_post_processing \n
#                   1/ Fill the desired options in the inputs/configuration.csv file \n
#                   2/ Run main_post_processing
#
##

##
# @package main_post_processing
# @author R. Douvenot
# @date 20/07/2021 (created) 13/12/22 (last modif)
# @version in progress
#
# @brief Computes and plots the demanded data from the final field and total wavelet parameters given by SSW
#
# @param[in]
# Inputs are defined in the files in the "inputs" directory
# - post_processing.csv that contains \n
# -- Data type:         Type of data to plot, may be 'E' (E-field), 'F' (propag factor), or 'S' (Poynting vector) \n
# -- Final field:       Plot the last calculation plane. 'Y' or 'N' \n
# -- Total field:       Plot the total field from the wavelet parameters. 'Y' or 'N' \n
# -- Dynamic:           Difference between the min and max value plotted (in dB) \n
#
# @param[out]
# Figures are saved in the "outputs" directory
#
# nota: the other parameters (frequency, wavelet, and so on) are read from the source and propagation modules
#
##

import scipy.constants as cst
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
import pywt
import sys
import numpy as np
from src.plots.plot_field import plot_field, plot_dictionary
# where config is defined
from src.classes_and_files.read_files import read_config, read_config_plot

# input: the source information
file_source_config = '../source/inputs/configuration.csv'
# input: main configuration file
file_configuration = '../propagation/outputs/configuration.csv'
# input: main configuration for plots
file_config_plot = './inputs/configuration.csv'

# --- Define and fill the config variable that contains all the simulation parameters --- #
config = read_config(file_configuration, file_source_config)
# --------------------------------------------------------------------------------------- #

# parameters of the plots
config_plot = read_config_plot(file_config_plot)

# plot the final field -- Vertical cut
plot_field(config, config_plot)

# plot the library
if config_plot.library == 'Y':
    plot_dictionary(config, config_plot)




