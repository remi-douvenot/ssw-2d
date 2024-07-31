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
        self.P = None
        self.Q = None
# ---------- END --------- #
# --- Defining classes --- #
# ------------------------ #


def read_config(file_configuration):
    # ----------------------------- #
    # --- Reading configuration --- #
    # ----------------------------- #
    config = Config()
    with open(file_configuration, newline='') as f_config:
        file_tmp = csv.reader(f_config)
        for row in file_tmp:
            if row[0] == 'N_x':
                config.N_x = int(row[1])
            elif row[0] == 'x_step':
                config.x_step = np.float64(row[1])
            elif row[0] == 'z_max_relief':
                config.z_max_relief = np.float64(row[1])
            elif row[0] == 'type':
                config.type = row[1]
            elif row[0] == 'iterations':
                config.iterations = int(row[1])
            elif row[0] == 'width':  # width of the triangle relief
                config.width = np.float64(row[1])
            elif row[0] == 'center':  # center of the triangle relief
                config.center = np.float64(row[1])
            elif row[0] == 'P':
                # Convert "43.76654;65.875675" to (43.76654, 65.875675) tuple of floats
                config.P = tuple([np.float64(l) for l in row[1].split(';')])
            elif row[0] == 'Q':
                # Same
                config.Q = tuple([np.float64(l) for l in row[1].split(';')])
            elif row[0] == 'Property':
                pass  # first line
            else:
                raise ValueError(['Input file of the configuration is not valid. Input "' + row[0] + '" not valid'])
    if config.type in ('Bing', 'IGN') and (config.P is None or config.Q is None):
        raise ValueError('P and Q must be set when Bing or IGN terrain type is selected.')

    # ------------ END ------------ #
    # --- Reading configuration --- #
    # ----------------------------- #
    return config
