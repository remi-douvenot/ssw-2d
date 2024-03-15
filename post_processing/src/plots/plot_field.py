# This file is part of SSW-2D.
# SSW-2D is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# SSW-2D is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
# of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Foobar. If not, see
# <https://www.gnu.org/licenses/>.

import numpy as np
import scipy.constants as cst
import pywt
import matplotlib.pyplot as plt
import matplotlib
from scipy.sparse import coo_matrix


# put relief below the field (upward shift of the field)
def shift_relief(u_field, ii_relief):
    if ii_relief == 0:
        u_field_shifted = u_field
    else:
        u_field_shifted = np.zeros_like(u_field)
        u_field_shifted[ii_relief:] = u_field[:-ii_relief]
    return u_field_shifted

##
# @package plot_field
# @author R. Douvenot
# @date 17/06/2022
# @version V1
#
# @brief plots the final and total fields from all the parameters (propagation and plot configurations)
#
# @param[in] config         Class that contains the propagation parameters (see Classes)
# @param[in] config_plot    Class that contains the plot parameters (see Classes)
#
# @param[out] None          Plots are displayed and saved in the "outputs" directory
##


# --- plot the source field --- #
def plot_field(config, config_plot):
    # --- Load wavelets along x --- #
    wv_total = np.load('../propagation/outputs/wv_total.npy', allow_pickle=True)
    # --- Load relief --- #
    z_relief = np.loadtxt('../terrain/outputs/z_relief.csv', delimiter=',', dtype="float")
    diff_relief = np.diff(z_relief)
    # --- Read the parameter values --- #
    n_x = config.N_x
    n_z = config.N_z
    wv_l = config.wv_L
    wv_family = config.wv_family
    x_s = - config.x_s
    x_max = config.N_x * config.x_step  # x_max in km
    x_step = config.x_step
    z_max = config.z_step * config.N_z
    z_step = config.z_step
    freq = config.freq
    k0 = 2 * cst.pi * freq / cst.c
    z_apo = int(config.apo_z * z_max)  # altitude of apodisation
    n_apo_z = int(np.round(n_z*config.apo_z))

    # --- Initialise field --- #
    u_field_total = np.zeros((n_x, n_z), dtype='complex')
    e_field_total = np.zeros((n_x, n_z), dtype='complex')

    wv_ii_x = [np.array([])] * (wv_l + 1)

    # --- Image layer --- #
    ground_type = config.ground
    if ground_type == 'None' or config.method == 'SSF':  # No ground, no image layer
        n_im = 0
    else:  # ground, therefore an image layer different from 0
        image_layer = config.image_layer  # image_layer in % of the total size n_z
        n_im = int(np.round(n_z * image_layer))
        remain_im = n_im % 2 ** wv_l
        if remain_im != 0:
            n_im += 2 ** wv_l - remain_im

    # --- from wavelets to E-field --- #
    # loop on each distance step
    for ii_x in np.arange(0, n_x):  # first field is not saved
        # from coo matrix to array on each level
        for ii_lvl in np.arange(0, wv_l + 1):
            wv_ii_x[ii_lvl] = wv_total[ii_x][ii_lvl].todense()
            wv_ii_x[ii_lvl] = np.array(wv_ii_x[ii_lvl])
        # inverse fast wavelet transform
        # squeeze to remove the first useless dimension
        uu_x = np.squeeze(pywt.waverec(wv_ii_x, wv_family, 'per'))
        # remove image field
        uu_x = uu_x[n_im:]
        # add the relief
        if ground_type == 'PEC' or ground_type == 'dielectric':
            # whether ascending or descending relief, the shift is made before or after free-space propagation
            if diff_relief[ii_x] < 0:
                ii_relief = int(z_relief[ii_x + 1] / z_step)
            else:
                ii_relief = int(z_relief[ii_x] / z_step)
            uu_x = shift_relief(uu_x, ii_relief)
        x_current = x_s + (ii_x + 1) * x_step
        # print('x_current', x_current)

        e_field_total[ii_x, :] = uu_x / np.sqrt(k0 * x_current) * np.exp(-1j * k0 * x_current)
    # -------------------------------- #

    # --- 2D plot --- #
    output_type = config_plot.output_type
    print('output_type', output_type)
    with np.errstate(divide='ignore'):  # ignore log10(0) warning
        # output = electric field
        if output_type == 'E':  # electric field, (dBV/m)':
            data_db = 20 * np.log10(np.abs(e_field_total)).T
            title = ['Electric field (dBV/m)']
            x_label = 'E (dBV/m)'
            # output = propagation factor
        elif output_type == 'F':
            # F = \sqrt{2\pi} E r / \sqrt{\zeta_0 P_Tx G_Tx}
            zeta_0 = 1 / (cst.epsilon_0 * cst.c)
            x_vect = np.arange(x_step, x_max + x_step, x_step) - x_s
            f_field_total = np.sqrt(2 * cst.pi / zeta_0) * e_field_total
            f_field_total *= np.array(x_vect)[:, np.newaxis]
            data_db = 20 * np.log10(np.abs(f_field_total)).T
            x_label = 'F (dB)'
        # output = Poynting vector
        elif output_type == 'S':
            # S = E^2 / \sqrt{2 \zeta_0}
            zeta_0 = 1 / (cst.epsilon_0 * cst.c)
            data_db = 20 * np.log10(np.abs(e_field_total) / (np.sqrt(2 * zeta_0))).T
            x_label = 'S (dBW/m2)'
        else:
            raise ValueError(['Not such an option !! o_O'])

    v_max = np.max(data_db)
    dynamic = config_plot.dynamic
    v_min = v_max - dynamic
    # geometry
    z_vect = np.linspace(0, z_step * n_z, n_z, endpoint=False)
    # plot relief in black
    x_vect = np.linspace(0, x_max, n_x + 1, endpoint=True) / 1000

    if config_plot.total_flag == 'Y':
        # clear the plot for updates
        fig = plt.figure(figsize=(8.2, 3.8), tight_layout=True)

        # plot the field
        limits = [x_step * 1e-3, x_max * 1e-3, 0, z_max]  # x in km, z in m
        im = plt.imshow(data_db, cmap='jet', aspect='auto', vmax=v_max, vmin=v_min, origin='lower',
                       interpolation='none', extent=limits)
        cb = plt.colorbar(im)
        cb.set_label(x_label, labelpad=-20, y=-0.05, rotation=0, fontsize=12)

        ax = im.axes

        ax.set_xlabel('Distance (km)', fontsize=12)
        ax.set_ylabel('Altitude (m)', fontsize=12)
        # ax.set_colorbar(jet)

        # --- Ground plot --- #
        ax.set_xlim((0, x_max / 1000))
        # If there is a ground
        if ground_type == 'PEC' or ground_type == 'Dielectric':
            # relief in black
            ax.plot(x_vect, z_relief, 'k')
            ax.fill_between(x_vect, z_relief, where=z_relief > 0, facecolor='black')
            # print('n_z', n_z, 'e_field_db size', data_db_final.size)
        elif ground_type == 'None':
            pass
        else:
            raise ValueError('Ground type not recognized (plots.py)')

        # --- Apodisation plot ("top" or "bottom + top") --- #
        if ground_type == 'PEC' or ground_type == 'Dielectric':
            ax.hlines(z_max - z_apo, 0, x_max, colors='k', linestyles='dotted')
        elif ground_type == 'None':
            ax.hlines([z_apo, z_max - z_apo], 0, x_max, colors='k', linestyles='dotted')

        # save
        fig.savefig('./outputs/total_field.png')

    if config_plot.final_flag == 'Y':
        # --- data to plot --- #
        data_db_final = data_db[:, -1]
        v_max = np.max(data_db_final) + 2
        v_min = v_max - dynamic
        # --- Final field plot --- #
        fig = plt.figure(figsize=(2.7, 3.8), tight_layout=True)
        ax = fig.add_subplot(111)
        plt.plot(data_db_final, z_vect)
        ax.set_xlim(v_min, v_max)
        ax.set_ylim(0, z_step * n_z)
        if output_type == 'E':
            x_label = 'E (dBV/m)'
        elif output_type == 'F':
            x_label = 'F (dB)'
        elif output_type == 'S':
            x_label = 'S (dBW/m2)'
        ax.set_xlabel(x_label, fontsize=12)
        # --- Apodisation plot ("top" or "bottom + top") --- #
        if ground_type == 'PEC' or ground_type == 'Dielectric':
            ax.hlines(z_max - z_apo, v_min, v_max, colors='k', linestyles='dotted')
        elif ground_type == 'None':
            ax.hlines([z_apo, z_max - z_apo], v_min, v_max, colors='k', linestyles='dotted')
        # ax.set_ylabel('Altitude (m)', fontsize=12)
        ax.grid('on')
        # print('Max value = ', np.round(v_max, 2))
        # save
        fig.savefig('./outputs/final_field.png')

    # Plot field and the wavelet coefficients wrt vertical at the desired horizontal distance
    if config_plot.wavelets == 'Y':
        # plot_field_cut(config, 57800, u_x, ii_x, n_apo_z)
        ii_x = int(np.round(config_plot.cut/config.x_step))
        title = 'Wavelets at x = ' + str(config_plot.cut/1000) + ' km'
        plot_wavelet_cut(config, config_plot.cut, wv_total[ii_x], z_max, dynamic, title)
        # plot_field_cut(config, 58600, u_x, ii_x, n_apo_z)
        # plot_wavelet_cut(config, 35000 - config.x_step, wv_total[ii_x], n_apo_z, n_im, z_max)
        # plot_wavelet_cut(config, 50000 - config.x_step, wv_total[ii_x], n_apo_z, n_im, z_max)

    plt.show()


##
# @package plot_wavelet_cut
# @author R. Douvenot
# @date 27/06/2022
# @version V1
#
# @brief Plots the (vertical) wavelet decomposition of the field for a given distance
#
# @param[in] config         Class that contains the propagation parameters (see Classes)
# @param[in] x_cut          Distance at which the cut is chosen
# @param[in] wv_x           The wavelet decomposition (a list of coo matrices)
# @param[in] z_max          Max altitude
# @param[in] dynamic        The dynamic (difference min - max)
# @param[in] title          Title of the figure
#
# @param[out] None          Plots are displayed and saved in the "outputs" directory
##

# Plot the WAVELET COEFFICIENTS on the desired cut
def plot_wavelet_cut(config, x_cut, wv_x, z_max, dynamic, title):
    print('ii_x = ', int(np.round(x_cut/config.x_step)))

    # PLOT THE WAVELETS
    n_z = config.N_z
    n_im = int(config.image_layer*n_z)

    # total coeffs
    coeffs_for_show = np.zeros([config.wv_L + 1, n_z + n_im])
    # coefs that will be plotted (without apodisation & image layer)
    coeffs_for_show2 = np.zeros([config.wv_L + 1, n_z])

    # Scaling function
    ll_level = 0  # phi: scaling function
    for ii in range(0, wv_x[ll_level].nnz):
        ii_2 = wv_x[ll_level].col[ii] * 2 ** config.wv_L
        value = abs(wv_x[ll_level].data[ii])
        coeffs_for_show[ll_level, ii_2:ii_2 + 2 ** config.wv_L] = value
    # remove image layer and apodisation
    coeffs_for_show2[ll_level, :] = 20 * np.log10(np.abs(coeffs_for_show[ll_level, n_im:]))
    n_wavelets = wv_x[ll_level].nnz
    # Wavelet levels
    for ll_level in range(1, config.wv_L + 1):
        for ii in range(0, wv_x[ll_level].nnz):
            ii_2 = wv_x[ll_level].col[ii] * 2 ** (config.wv_L + 1 - ll_level)
            value = abs(wv_x[ll_level].data[ii])
            coeffs_for_show[ll_level, ii_2:ii_2 + 2 ** (config.wv_L + 1 - ll_level)] = value
        # remove image layer and apodisation
        coeffs_for_show2[ll_level, :] = 20 * np.log10(np.abs(coeffs_for_show[ll_level, n_im:]))
        n_wavelets += wv_x[ll_level].nnz

    # fig = plt.figure(figsize=(3, 6))
    fig = plt.figure(figsize=(4.0, 5.0), tight_layout=True)
    plt.title(title)
    # ax = fig.add_subplot(1, 1, 1)
    # ax.tick_params(labelsize=12)
    v_max = np.max(coeffs_for_show2)
    # v_min = 0
    my_cmap = matplotlib.cm.get_cmap('jet').copy()
    # my_cmap.set_under('w')
    z_min_plot = 0  # choose min altitude to display
    z_max_plot = z_max  # choose max altitude to display
    l_index = int(z_min_plot / config.z_step)
    r_index = int(z_max_plot / config.z_step)
    # print(l_index, r_index)
    # print('v_max', np.max((coeffs_for_show2[:, l_index:r_index])))

    im = plt.imshow(coeffs_for_show2[:, r_index:l_index:-1].transpose(),
                    extent=[-0.5, config.wv_L + 1, z_min_plot, z_max_plot], cmap=my_cmap,
                    interpolation='none', aspect='auto', vmax=v_max, vmin=v_max - dynamic)
    cb = plt.colorbar(im)
    cb.set_label(label='Coef \n magn (dB)', labelpad=-20, y=-0.05, rotation=0, fontsize=12)
    # cb.ax.tick_params(labelsize=10)
    col_labels = ['phi' + str(config.wv_L)]
    for ii in np.arange(0, config.wv_L):
        col_labels.append('psi' + str(config.wv_L-ii))
    # col_labels = [0, 1, 2, 3, 4]
    im.axes.set_xticks(np.arange(coeffs_for_show2.shape[0] + 1) - .5)
    plt.xticks(np.arange(0, config.wv_L+1, 9/8))
    im.axes.set_xticklabels(col_labels)
    # tick.label1.set_horizontalalignment('center')
    plt.xlabel("Decomposition level", fontsize=12)
    plt.ylabel("Altitude (m)", fontsize=12)
    # im.axes.xaxis.label_position = 'center'
    for tick in im.axes.xaxis.get_major_ticks():
        # tick.tick1line.set_markersize(0)
        tick.label1.set_horizontalalignment('center')
    print('Compression rate = ', (1-n_wavelets/config.N_z)*100, ' %')

    # # Horizontal lines
    # for ii in np.arange(0, z_max/config.z_step, 2**config.wv_L):
    #     plt.hlines(ii*config.z_step, -0.5, config.wv_L + 1, linestyles='--', linewidth=1)

    # Vertical lines
    for ii in np.arange(0, config.wv_L+1, 9/8)+10/16:
        plt.vlines(ii, 0, z_max, linestyles='--', linewidth=1, colors='k')

    # save figure
    fig.savefig('./outputs/'+title+'.png')

    plt.show()


##
# @package plot_dictionary
# @author R. Douvenot
# @date 27/06/2022
# @version V1
#
# @brief Plots each wavelet decomposition of the propagation dictionary
#
# @param[in] config         Class that contains the propagation parameters (see Classes)
# @param[in] config_plot    Class that contains the plot parameters (see Classes)
#
# @param[out] None          Plots are displayed and saved in the "outputs" directory
##

# Plot the WAVELET LIBRARY
def plot_dictionary(config, config_plot):

    # download dictionary
    dictionary = np.load('../propagation/outputs/dictionary.npy', allow_pickle=True)
    # max level of decomposition
    ll = len(dictionary) - 1
    # max altitude = number of fastest wavelet coefficients * 2 * Delta z
    z_max = len(dictionary[0][0][-1])*2*config.z_step
    # no image layer
    config.image_layer = 0
    # make it sparse

    # on each level
    for ii in np.arange(0, ll+1):

        for ii_q in np.arange(0, len(dictionary[ii])):
            # fill the wavelet levels
            dictionary_coo = [[] for _ in range(0, ll+1)]
            for ii_lvl in np.arange(0, ll + 1):

                dictionary_coo[ii_lvl] = coo_matrix(dictionary[ii][ii_q][ii_lvl])

            if ii == 0:
                title = 'Scaling function phi' + str(ll)
            else:
                title = 'Wavelet psi' + str(ll-ii+1) + ', number ' + str(ii_q+1) + ' out of ' + str(2**(ii-1))
            plot_wavelet_cut(config, 0, dictionary_coo, z_max, config_plot.dynamic, title)

            # plot the equivalent electric field
            wavelet_field = pywt.waverec(dictionary[ii][ii_q], config.wv_family, mode='per')
            # geometry
            fig = plt.figure(figsize=(3.0, 5.0), )
            n_z = wavelet_field.size
            z_vect = np.linspace(0, config.z_step * n_z, n_z, endpoint=False)
            wavelet_db = 20 * np.log(np.abs(wavelet_field))
            plt.plot(wavelet_db, z_vect)
            plt.xlim(wavelet_db.max()-config_plot.dynamic, wavelet_db.max())
            plt.grid('on')
            plt.ylabel('Altitude (m)', fontsize=12)
            plt.xlabel('E (dBV/m)', fontsize=12)
            plt.tight_layout()  # pad=0.4, w_pad=0.5, h_pad=1.0)
