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
# @date 11/01/2022
# @brief Fill the config class with the input from input file
##

# where config class is defined
import csv
import numpy as np


# ------------------------ #
# --- Defining classes --- #
# ------------------------ #
class Config:
    def __init__(self):
        self.N_x = 0
        self.x_step = 0.0
        self.type = 'None'
        self.iterations = 0
        self.z_max_relief = 0.0
        self.center = 0.0
        self.width = 0.0
# ---------- END --------- #
# --- Defining classes --- #
# ------------------------ #


def read_config(file_configuration):
    # ----------------------------- #
    # --- Reading configuration --- #
    # ----------------------------- #
    f_config = open(file_configuration, newline='')
    file_tmp = csv.reader(f_config)
    for row in file_tmp:
        if row[0] == 'N_x':
            Config.N_x = int(row[1])
        elif row[0] == 'x_step':
            Config.x_step = float(row[1])
        elif row[0] == 'z_max_relief':
            Config.z_max_relief = float(row[1])
        elif row[0] == 'type':
            Config.type = row[1]
        elif row[0] == 'iterations':
            Config.iterations = int(row[1])
        elif row[0] == 'width':  # width of the triangle relief
            Config.width = float(row[1])
        elif row[0] == 'center':  # center of the triangle relief
            Config.center = float(row[1])
        elif row[0] == 'Property':
            pass  # first line
        else:
            raise ValueError(['Input file of the configuration is not valid. Input "' + row[0] + '" not valid'])

    # ------------ END ------------ #
    # --- Reading configuration --- #
    # ----------------------------- #

    return Config
