# This file is part of SSW-2D.
# SSW-2D is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# SSW-2D is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
# of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Foobar. If not, see
# <https://www.gnu.org/licenses/>.

# ------------------------ #
# --- Defining classes --- #
# ------------------------ #
class Config:
    def __init__(self):
        self.N_x = 0
        self.N_y = 0
        self.N_z = 0
        self.x_step = 0
        self.y_step = 0
        self.z_step = 0
        self.x_s = 0 # distance of the source (negative value)
        self.freq = 0
        self.max_compression_err = 0 # Max compression error
        self.V_s = 0 # compression threshold on signal
        self.V_p = 0 # compression threshold on propagator
        self.wv_family = 'None'
        # wavelet level
        self.wv_L = 0
        # type of apodisation window
        self.apo_window = 'None'
        # percentage of apodisation of the domain along y
        self.apo_y = 0
        # percentage of apodisation of the domain along z
        self.apo_z = 0
        # percentage of image layer of the domain along z (if any ground)
        self.image_layer = 0
        # number of point sin the image layer (multiple of 2^L)
        self.N_im = 0
        # ground type ('None', 'PEC', or 'dielectric')
        self.ground = 'None'
        # ground relative permittivity (for dielectric ground only)
        self.epsr = 0
        # ground conductivity (for dielectric ground only)
        self.sigma = 0
        self.atmosphere = 'None'  # atmospheric profile type
        self.c0 = 0  # standard atm gradient
        self.delta = 0  # evaporation duct height
        self.zb = 0  # base height of a trilinear duct
        self.c2 = 0  # gradient in a trilinear duct
        self.zt = 0  # thickness of a trilinear duct
        self.atm_filename = 'None'  # file for a hand-generated atmospheric profile
        self.turbulence = 'N'
        self.Cn2 = 0
        self.L0 = 0

class Config_plot:
    def __init__(self):
        self.dynamic = 0
        self.output_type = 'None'
        self.final_flag = 'None'
        self.total_flag = 'None'
# ---------- END --------- #
# --- Defining classes --- #
# ------------------------ #
