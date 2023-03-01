##
# @brief function that computes and adds the image field for the local image method.
# @author R. Douvenot
# @package compute_image_field
# @date 02/09/21
# @version OK
#
# @details function that compute and add the image field for the local image method. The image field is computed with
# Fresnel coefficient and added on N_im point
# u_x_im = compute_image_field(u_x,ground_coeff,simulation_parameters,wavelet_parameters)
#
# @params[in] u_x : reduced electric field (complex array)
# @params[in] config : class with the parameters
# @params[in] N_im : size of the image layer
# @params[out] u_x_im : reduced field with the image layer added
##


import numpy as np


def compute_image_field(u_x, n_im):

    # compute the image field on N_im point for the image theorem
    # print('compute the image field on N_im point for the image theorem')

    # Simulation parameters
    n_z = u_x.size

    # Init the size of the field added with image layer
    u_x_im = np.zeros([n_z + n_im], dtype='complex')

    # Fill the image layer
    u_x_im[n_im+1:n_z + n_im] = u_x[1:n_z]
    u_x_im[0:n_im] = - u_x[n_im:0:-1]
    return u_x_im

##
# @brief function that computes and adds the image field for the local image method.
# @author R. Douvenot
# @package compute_image_field_TM_PEC
# @date 02/05/22
# @version WIP
#
# @details function that compute and add the image field for the local image method. The image field is computed with
# Fresnel coefficient +1 and added on N_im point
# This function is only useful for a TM polar on a PEC.
# u_x_im = compute_image_field(u_x,ground_coeff,simulation_parameters,wavelet_parameters)
#
# @params[in] u_x : reduced electric field (complex array)
# @params[in] config : class with the parameters
# @params[in] N_im : size of the image layer
# @params[out] u_x_im : reduced field with the image layer added
##


def compute_image_field_tm_pec(u_x, n_im):

    # compute the image field on N_im point for the image theorem
    # print('compute the image field on N_im point for the image theorem')

    # Simulation parameters
    n_z = u_x.size

    # Init the size of the field added with image layer
    u_x_im = np.zeros([n_z + n_im], dtype='complex')

    # Fill the image layer
    u_x_im[n_im:n_z + n_im] = u_x
    u_x_im[0:n_im] = u_x[n_im:0:-1]
    u_toto = np.abs(u_x_im)
    return u_x_im

