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
# @package aperture_field
# @version work in progress. Use with care.
# @brief Function that plot the field of an aperture source (with uniform law on the aperture for now)
##

#######################################################################################################################
# function : init_aperture_field
# author : T. Bonnafont
# Date : 04/12/18 / Last modif : 12/12/18
# State : OK
#
# function that plot the field of an aperture source (with uniform law on the aperture for now)
# u_field = init_aperture_field(simulation_parameters,source_parameters)
#
# WARNING : axis y is between -y_max and y_max
#
# INPUTS :
# - simulation_parameters : Structure. Class containing the simulation parameters
# - source_parameters : Structure. Class containing the source parameters (law and size for the aperture)
#
# OUTPUT :
# - u_field : (N_y,N_z)-array. Contains the field of an aperture with source position
#######################################################################################################################

import numpy as np
from scipy.fftpack import fftshift,fft

def init_aperture_field(x0,x_s,y_s,z_s,simulation_parameters,source_parameters):
    ####################################### PARAMETERS #################################################################
    #--------------- simulation parameters ----------------------------------------------------------------------------#
    N_y = simulation_parameters.N_y
    step_y = simulation_parameters.y_step
    N_z = simulation_parameters.N_z
    step_z = simulation_parameters.z_step
    x0 = x0
    k0 = simulation_parameters.k0

    #--------------- source parameters --------------------------------------------------------------------------------#
    x_s = x_s
    y_s = y_s
    z_s = z_s
    w_y = source_parameters.w_y
    w_z = source_parameters.w_z
    # number of point for the source position and width
    N_y_s = int(y_s / step_y)
    N_z_s = int(z_s / step_z)
    n_w_y = int(w_y / step_y)
    n_w_z = int(w_z / step_z)
    # centered the source between -y_max and y_max on y axis
    # N_y_s += int(N_y/2)
    ########################## COMPUTE THE FIELD OF THE APERTURE USING PLANE WAVE SPECTRUM #############################
    # Init the field
    u_field = np.zeros((N_y,N_z),dtype='complex')

    # Compute the position of the first iteration from the source
    x_pos = x0 - x_s

    # Compute the aperture law
    for ii_y in np.arange(max(0,N_y_s-n_w_y),min(N_y_s+n_w_y,N_y)):
        for ii_z in np.arange(max(0,N_z_s-n_w_z),min(N_z_s+n_w_z,N_z)):
            if source_parameters.aperture_law == 'Uniform':
                u_field[ii_y,ii_z] = 1.0
            else :
                raise('ERROR NOT CODED YET')

    # If the first iteration is at the source position then the field correspond to the aperture law
    if x_pos == 0 :
        u_field = u_field

    # If the first iteration is beyond the position of the source then we do the fourier transform of the aperture and
    # propagate using the plane wave sprectrum
    else:
        # Fourier transform of the aperture
        U_field = fftshift(fft(fft(u_field,axis=-1),axis=0)) # centered on z axis on the z position of the source
        # Propagate with plane wave spectrum : valid at far field
        # loop over the y direction
        for ii_y in np.arange(0, N_y):
            # compute the y position from the source
            y_pos = step_y * ii_y - y_s
            # loop over the z direction
            for ii_z in np.arange(0, N_z):
                # compute the z position from the source
                z_pos = step_z * ii_z - z_s
                # compute the range from the source
                rr_2 = x_pos ** 2 + y_pos ** 2 + z_pos ** 2 + 1e-15
                rr = np.sqrt(rr_2)
                # compute the angle
                cos_theta = x_pos / rr
                u_field[ii_y,ii_z] = 1j*k0*cos_theta*U_field[ii_y,ii_z]*np.exp(-1j*k0*rr)/rr

    # normalised the field
    u_field *= 1/np.max(np.abs(u_field))
    normalisation_factor = np.max(np.abs(u_field))
    return u_field, normalisation_factor