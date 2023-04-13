# This file is part of SSW-2D.
# SSW-2D is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# SSW-2D is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
# of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Foobar. If not, see
# <https://www.gnu.org/licenses/>.

import scipy.constants as cst
import numpy as np
import pywt
import warnings
import matplotlib.pyplot as plt


class Plots(object):

    # --- plot the source field in the GUI --- #
    def plot_source_in(self):
        ax = self.ax_source  # defined in the main
        ax.clear()
        self.plot_source(ax)
        self.canvas_source.draw()

    # --- plot the source field as an external figure --- #
    def plot_source_out(self):
        fig = plt.figure(figsize=(2.6, 3.8), tight_layout=True)
        ax = fig.add_subplot(111)
        self.plot_source(ax)
        fig.show()

    # --- plot the source field --- #
    def plot_source(self, ax):
        # ignore the "divide by zero" warning for plots
        np.seterr(divide='ignore')

        # E field
        e_field = np.load('../source/outputs/E_field.npy')
        e_field_db = 20 * np.log10(np.abs(e_field))
        v_max = np.max(e_field_db)
        dynamic = self.dynamicSpinBox.value()
        v_min = v_max - dynamic

        # geometry
        z_step = self.deltaZMDoubleSpinBox.value()
        n_z = self.nZSpinBox.value()
        z_vect = np.linspace(0, z_step*n_z, n_z, endpoint=False)
        # print('n_z', n_z, 'e_field_db size', e_field_db.size)
        # Relief: initial field is shifted upwards by the relief at z=0
        z_relief = np.loadtxt('../terrain/outputs/z_relief.csv', delimiter=',', dtype="float")

        # plot
        ground_type = self.groundTypeComboBox.currentText()
        # if ground = None, no shift due to relief
        z_relief_init = 0
        # else, we shift the source and plot the ground
        if ground_type == 'PEC' or ground_type == 'Dielectric':
            z_relief_init += z_relief[0]
            ax.hlines(z_relief[0], v_min, v_max, colors='k')
        ax.plot(e_field_db, z_vect+z_relief_init)  # field shifted of the value of the ground at the origin

        ax.set_xlim(v_min, v_max)
        ax.set_ylim(0, z_step*n_z)
        ax.set_xlabel('E field (dBV/m)', fontsize=12)
        ax.set_ylabel('Altitude (m)', fontsize=12)
        ax.grid('on')
        # @TODO Print the max value somewhere on the interface
        # print('Max field = ', np.round(v_max, 2), 'dBV/m')

    # --- plot the field in the GUI --- #
    def plot_field_in(self):
        # prepare axes
        self.ax_field.clear()
        self.ax_final.clear()
        ax_field = self.ax_field  # defined in the main
        ax_final = self.ax_final  # defined in the main
        # define plots
        self.plot_field(ax_field, ax_final)
        # display plots
        self.canvas_field.draw()
        self.canvas_final.draw()

    # --- plot the 2D field map in external figures --- #
    def plot_field_out(self):
        fig_field = plt.figure(figsize=(8.2, 3.8), tight_layout=True)
        fig_final = plt.figure(figsize=(2.2, 3.8), tight_layout=True)
        ax_field = fig_field.add_subplot(111)
        ax_final = fig_final.add_subplot(111)
        # define plots
        self.plot_field(ax_field, ax_final)
        # display plots
        fig_field.show()

    # --- plot the final field in external figures --- #
    def plot_final_out(self):
        fig_field = plt.figure(figsize=(8.2, 3.8), tight_layout=True)
        fig_final = plt.figure(figsize=(2.2, 3.8), tight_layout=True)
        ax_field = fig_field.add_subplot(111)
        ax_final = fig_final.add_subplot(111)
        # define plots
        self.plot_field(ax_field, ax_final)
        # display plots
        fig_final.show()

    # --- plot the source field --- #
    # TODO : Remove coefficients corresponding to image field in the saved wavelets in SSW
    def plot_field(self, ax_field, ax_final):
        # --- Load wavelets along x --- #
        wv_total = np.load('../propagation/outputs/wv_total.npy', allow_pickle=True)
        # --- Load relief --- #
        z_relief = np.loadtxt('../terrain/outputs/z_relief.csv', delimiter=',', dtype="float")
        diff_relief = np.diff(z_relief)
        # --- Read the parameter values --- #
        n_x = self.nXSpinBox.value()
        n_z = self.nZSpinBox.value()
        wv_l = self.wvlMaxLevelSpinBox.value()
        wv_family = self.wvlFamilyComboBox.currentText()
        x_s = - self.x_sDoubleSpinBox.value()
        x_max = self.xMaxKmDoubleSpinBox.value()*1e3  # x_max in km
        x_step = self.deltaXMDoubleSpinBox.value()
        z_max = self.zMaxMDoubleSpinBox.value()
        z_step = self.deltaZMDoubleSpinBox.value()
        freq = self.frequencyMHzDoubleSpinBox.value()
        k0 = 2*cst.pi*freq*1e6/cst.c
        z_apo = int(self.sizeApoSpinBox.value()/100 * z_max)  # altitude of apodisation
        n_im = int(self.sizeImageSpinBox.value()/100 * n_z)  # size of image layer
        method = self.methodComboBox.currentText()

        # --- Initialise field --- #
        e_field_total = np.zeros((n_x, n_z), dtype='complex')

        wv_ii_x = [[]] * (wv_l + 1)

        # --- Image layer --- #
        ground_type = self.groundTypeComboBox.currentText()
        if ground_type == 'None' or method == 'SSF':  # No ground or SSF -> no image layer
            n_im = 0
        else:  # n_im must be multiple of 2^L
            remain_im = n_im % 2 ** wv_l
            if remain_im != 0:
                n_im += 2 ** wv_l - remain_im

        # --- from wavelets to E-field --- #
        # loop on each distance step
        for ii_x in np.arange(0, n_x):  # first field is not saved
            # from coo matrix to array on each level
            for ii_lvl in np.arange(0, wv_l + 1):
                wv_ii_x[ii_lvl] = wv_total[ii_x][ii_lvl].todense()
            # inverse fast wavelet transform
            # squeeze to remove the first useless dimension
            uu_x = np.squeeze(pywt.waverec(wv_ii_x, wv_family, 'per'))
            # remove image field
            uu_x = uu_x[n_im:]
            # add the relief
            if ground_type == 'PEC' or ground_type == 'Dielectric':
                # whether ascending or descending relief, the shift is made before or after propagation
                if diff_relief[ii_x] < 0:
                    ii_relief = int(z_relief[ii_x+1]/z_step)
                else:
                    ii_relief = int(z_relief[ii_x] / z_step)
                uu_x = shift_relief(uu_x, ii_relief)
            x_current = -x_s + (ii_x + 1) * x_step
            e_field_total[ii_x, :] = uu_x / np.sqrt(k0 * x_current) * np.exp(-1j * k0 * x_current)
        # -------------------------------- #

        # --- 2D plot --- #
        # @todo output = Take into account P_tx and G_Tx
        output_type = self.outputComboBox.currentText()
        # print('output_type', output_type)
        with np.errstate(divide='ignore'):  # ignore log10(0) warning
            # output = electric field
            if output_type == 'E (dBV/m)':
                data_db = 20 * np.log10(np.abs(e_field_total)).T
                title = ['Electric field (dBV/m)']
            # output = propagation factor
            elif output_type == 'F (dB)':
                # F = \sqrt{2\pi} E r / \sqrt{\zeta_0 P_Tx G_Tx}
                zeta_0 = 1/(cst.epsilon_0 * cst.c)
                x_vect = np.arange(x_step, x_max+x_step, x_step) - x_s
                f_field_total = np.sqrt(2 * cst.pi / zeta_0) * e_field_total
                f_field_total *= np.array(x_vect)[:, np.newaxis]
                data_db = 20 * np.log10(np.abs(f_field_total)).T
            # output = Poynting vector
            elif output_type == 'S (dBW/m2)':
                # S = E^2 / \sqrt{2 \zeta_0}
                zeta_0 = 1/(cst.epsilon_0 * cst.c)
                data_db = 20 * np.log10(np.abs(e_field_total)/(np.sqrt(2*zeta_0))).T
            else:
                raise ValueError(['Not such an option !! o_O'])

        v_max = np.max(data_db)
        dynamic = self.dynamicSpinBox.value()
        v_min = v_max - dynamic

        # clear the plot for updates
        ax = ax_field  # defined in the input
        ax.clear()
        try:
            self.cb.remove()
        except:
            pass

        # plot the field
        limits = [x_step * 1e-3, x_max * 1e-3, 0, z_max]  # x in km, z in m
        im = ax.imshow(data_db, cmap='jet', aspect='auto', vmax=v_max, vmin=v_min, origin='lower',
                       interpolation='none', extent=limits)
        self.cb = plt.colorbar(im, ax=self.ax_field)
        self.cb.set_label(output_type, labelpad=-20, y=-0.05, rotation=0, fontsize=12)

        ax.set_xlabel('Distance (km)', fontsize=12)
        ax.set_ylabel('Altitude (m)', fontsize=12)
        # ax.set_colorbar(jet)
        self.canvas_field.draw()

        # --- data to plot --- #
        data_db_final = data_db[:, -1]
        v_max = np.max(data_db_final)+2
        v_min = v_max - dynamic
        # geometry
        z_vect = np.linspace(0, z_step * n_z, n_z, endpoint=False)
        # plot relief in black
        x_vect = np.linspace(0, x_max, n_x+1, endpoint=True)/1000
        ax.set_xlim((0, x_max/1000))
        # ax = plt.subplot(111)

        # --- Ground plot --- #
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
            ax.hlines(z_max-z_apo, 0, x_max, colors='k', linestyles='dotted')
        elif ground_type == 'None':
            ax.hlines([z_apo, z_max - z_apo], 0, x_max, colors='k', linestyles='dotted')

        # ax.vlines(35, 0, z_max, colors='k', linestyles='dotted')
        # ax.vlines(35.2, 0, z_max, colors='k', linestyles='dotted')

        # --- Final field plot --- #
        ax = ax_final
        ax.clear()  # defined in the input
        ax.plot(data_db_final, z_vect)
        ax.set_xlim(v_min, v_max)
        ax.set_ylim(0, z_step * n_z)
        ax.set_xlabel(output_type, fontsize=12)
        # --- Apodisation plot ("top" or "bottom + top") --- #
        if ground_type == 'PEC' or ground_type == 'Dielectric':
            ax.hlines(z_max-z_apo, v_min, v_max, colors='k', linestyles='dotted')
        elif ground_type == 'None':
            ax.hlines([z_apo, z_max - z_apo], v_min, v_max, colors='k', linestyles='dotted')
        # ax.set_ylabel('Altitude (m)', fontsize=12)
        ax.grid('on')
        # print('Max value = ', np.round(v_max, 2))

        self.canvas_final.draw()

    # --- plot the environment (relief and atmosphere) in the GUI --- #
    def plot_environment_in(self):
        ax = self.ax_environment  # defined in the main
        self.plot_environment(ax)
        self.canvas_environment.draw()

    # --- plot the environment (relief and atmosphere) in an external figure --- #
    def plot_environment_out(self):
        fig = plt.figure(figsize=(8.2, 2.5), tight_layout=True)
        ax = fig.add_subplot(111)
        self.plot_environment(ax)
        fig.show()

    # --- plot the atmosphere only in an external figure --- #
    def plot_refractivity_out(self):

        fig = plt.figure(figsize=(2.2, 3.8), tight_layout=True)
        ax = fig.add_subplot(111)
        # --- Read the geometry --- #
        n_z = self.nZSpinBox.value()
        z_max = self.zMaxMDoubleSpinBox.value()
        z_step = self.deltaZMDoubleSpinBox.value()

        # geometry vectors
        z_vect = np.linspace(0, z_max, n_z, endpoint=False)
        atm_type = self.atmTypeComboBox.currentText()
        m_0 = 330

        # choice of the atmospheric profile type
        if atm_type == 'Homogeneous':  # constant vacuum
            n_refractivity = np.zeros(n_z) + m_0
        elif atm_type == 'Standard':  # constant slope
            z_step = self.deltaZMDoubleSpinBox.value()
            c0 = self.c0DoubleSpinBox.value()  # c0 in M-unit/m
            n_refractivity = standard_atmosphere(n_z, z_step, c0)
        elif atm_type == 'Evaporation':  # log-linear profile
            z_step = self.deltaZMDoubleSpinBox.value()  # in m
            c0 = self.c0DoubleSpinBox.value()  # c0 in M-unit/m
            delta = self.deltaDoubleSpinBox.value()  # delta m
            n_refractivity = evaporation_duct(n_z, z_step, c0, delta)
        elif atm_type == 'Bilinear' or atm_type == 'Trilinear':  # bilinear or trilinear profile
            z_step = self.deltaZMDoubleSpinBox.value()  # in m
            c0 = self.c0DoubleSpinBox.value()  # c0 in M-unit/m
            if atm_type == 'Bilinear':  # bilinear = trilinear such as zb = 0
                zb = 0
            else:
                zb = self.zbDoubleSpinBox.value()  # zb in m
            c2 = self.c2DoubleSpinBox.value()  # c2 in M-unit/m
            zt = self.ztDoubleSpinBox.value()  # zt in m
            n_refractivity = trilinear_duct(n_z, z_step, c0, zb, c2, zt)
        else:
            raise ValueError(['Wrong atmospheric type!'])

        # plot refractivity
        ax.plot(n_refractivity, z_vect, color='b', linewidth=2)
        # plot atmosphere on relief
        ax.set_ylabel('Altitude (m)', fontsize=12)
        ax.set_xlabel('M index', fontsize=12)
        ax.set_ylim(0, z_max)
        ax.grid('on')
        fig.show()

    # --- plot the environment (relief and atmosphere) --- #
    def plot_environment(self, ax):
        # @TODO: remove relief when no ground

        # --- Read the geometry --- #
        n_x = self.nXSpinBox.value()
        n_z = self.nZSpinBox.value()
        x_max = self.xMaxKmDoubleSpinBox.value()  # x_max in km
        x_step = self.deltaXMDoubleSpinBox.value()
        z_max = self.zMaxMDoubleSpinBox.value()
        z_step = self.deltaZMDoubleSpinBox.value()
        ground_type = self.groundTypeComboBox.currentText()
        atm_type = self.atmTypeComboBox.currentText()
        m_0 = 330

        # geometry vectors
        z_vect = np.linspace(0, z_max, n_z, endpoint=False)
        x_vect = np.linspace(0, x_max, n_x+1, endpoint=True)

        # --- The relief --- #
        z_relief = np.loadtxt('../terrain/outputs/z_relief.csv', delimiter=',', dtype="float")
        n_x_relief = z_relief.size
        # size of the relief must match the geometry parameter n_x+1 (from 0 to n_x included)
        if n_x+1 != n_x_relief:
            raise ValueError(['Terrain is not compatible with geometry \n'
                              ' Length ', n_x_relief, 'instead of ', n_x+1, '. Please generate terrain again.'])

        # choice of the atmospheric profile type
        if atm_type == 'Homogeneous':  # constant vacuum
            n_refractivity = np.zeros(n_z)+m_0
        elif atm_type == 'Linear':  # constant slope
            z_step = self.deltaZMDoubleSpinBox.value()
            c0 = self.c0DoubleSpinBox.value()  # c0 in M-unit/m
            n_refractivity = standard_atmosphere(n_z, z_step, c0)
        elif atm_type == 'Evaporation':  # log-linear profile
            z_step = self.deltaZMDoubleSpinBox.value()  # in m
            c0 = self.c0DoubleSpinBox.value()  # c0 in M-unit/m
            delta = self.deltaDoubleSpinBox.value()  # delta m
            n_refractivity = evaporation_duct(n_z, z_step, c0, delta)
        elif atm_type == 'Trilinear' or atm_type == 'Bilinear':  # Bi- or Trilinear duct
            z_step = self.deltaZMDoubleSpinBox.value()  # in m
            c0 = self.c0DoubleSpinBox.value()  # c0 in M-unit/m
            c2 = self.c2DoubleSpinBox.value()  # c2 in M-unit/m
            if atm_type == 'Bilinear':
                zb = 0
            else:
                zb = self.zbDoubleSpinBox.value()  # zb in m
            zt = self.ztDoubleSpinBox.value()  # zb in m
            n_refractivity = trilinear_duct(n_z, z_step, c0, zb, c2, zt)
        else:
            raise ValueError(['Wrong atmospheric type!'])

        # plot relief
        ax.clear()
        # ax = self.figure2.Axes
        if ground_type != 'None':
            ax.plot(x_vect, z_relief, color='k', linewidth=2, label='relief')
            z_relief_1 = z_relief[int(n_x / 4)]
            z_relief_2 = z_relief[int(n_x / 2)]
            z_relief_3 = z_relief[int(3 * n_x / 4)]
        else:
            z_relief_1 = 0
            z_relief_2 = 0
            z_relief_3 = 0
        # ax.fill_between(x_relief, z_relief, where=z_relief > 0, facecolor='black')
        # refractivity rescaled for visualisation (with respect to x_max)
        n_refractivity_plot = (n_refractivity-n_refractivity[0]) * x_max * 2e-4
        # plotted at 3 positions along propagation
        ax.plot(x_max/4+n_refractivity_plot, z_vect + z_relief_1, color='b', label='refraction')
        ax.plot(x_max/2+n_refractivity_plot, z_vect+z_relief_2, color='b', label=None)
        ax.plot(3*x_max/4+n_refractivity_plot, z_vect+z_relief_3, color='b', label=None)
        ax.legend()

        # plot atmosphere on relief
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, z_max)
        ax.set_ylabel('Altitude (m)', fontsize=12)
        ax.set_xlabel('Distance (km)', fontsize=12)
        ax.grid('on')

#
# @package: eliminate_top_field
# @author: R. Douvenot
# @date: 07/09/2021
# @version: V1.0
#
# @brief Eliminate the top field in the apodisation layer due to periodic decomposition
# @param[in] u_x Field
# @param[out] u_x Field with top wavelets = 0
#
#
# def eliminate_top_field(u_x):
#
#     # find the last zero
#     zeros_indices = np.where(u_x == 0)[0]  # [0] because where gives a 1-dimensional tuple
#     # print(len(zeros_indices))
#     if len(zeros_indices) == 0:
#
#         warnings.warn("You should consider a higher apodisation domain")
#         ii_zero = u_x.size
#     else:
#         # print(zeros_indices)
#         # fill zeros up to this last value
#         ii_zero = np.max(zeros_indices)
#     u_x[ii_zero:-1] = 0
#
#     return u_x


# ------------------- #
# --- ATMOSPHERES --- #
# ------------------- #

# standard atmosphere
def standard_atmosphere(n_z, z_step, c0):
    # vector of vertical positions
    z_vect = np.linspace(0, z_step*n_z, n_z, endpoint=False)
    # refractivity
    # mean M0 is 330 at the ground level
    n_refractivity = 330 + z_vect*c0
    return n_refractivity


# evaporation duct
def evaporation_duct(n_z, z_step, c0, delta):
    # vector of vertical positions
    z_vect = np.linspace(0, z_step*n_z, n_z, endpoint=False)
    n_refractivity = np.zeros_like(z_vect)
    # z0
    z0 = 1.5e-4
    # separating in and above the duct (where renders tuples)
    indices_z_inf = np.where(z_vect <= 2*delta)
    indices_z_sup = np.where(z_vect > 2*delta)
    # print(indices_z_inf)
    # in the duct: refractivity following Paulus-Jeske model
    n_refractivity[indices_z_inf] = 330 + 0.125*(z_vect[indices_z_inf]
                                                 - delta*np.log((z_vect[indices_z_inf]+z0) / z0))
    # above the duct: standard atm
    n_refractivity[indices_z_sup] = 330 + z_vect[indices_z_sup]*c0
    # ensuring continuity
    n_refractivity[indices_z_inf] = n_refractivity[indices_z_inf] - n_refractivity[indices_z_inf[0][-1]] \
                                    + n_refractivity[indices_z_sup[0][0]] - c0*z_step

    return n_refractivity


# trilinear duct
def trilinear_duct(n_z, z_step, c0, zb, c2, zt):
    # vector of vertical positions
    z_vect = np.linspace(0, z_step*n_z, n_z, endpoint=False)
    n_refractivity = np.zeros_like(z_vect)
    # z0
    z0 = 1.5e-4
    # print(indices_z_inf)
    # below the duct
    indices_zb = np.where(z_vect <= zb)  # below the duct
    n_refractivity[indices_zb] = 330 + z_vect[indices_zb]*c0
    # in the duct
    indices_zt = np.where(z_vect > zb)
    n_refractivity[indices_zt] = 330 + zb * c0 + (z_vect[indices_zt]-zb) * c2
    # above the duct
    indices_above = np.where(z_vect > zt+zb)  # above the duct
    n_refractivity[indices_above] = 330 + zb * c0 + zt * c2 + (z_vect[indices_above]-zb-zt) * c0

    return n_refractivity


# put relief below the field (upward shift of the field)
def shift_relief(u_field, ii_relief):
    if ii_relief == 0:
        u_field_shifted = u_field
    else:
        u_field_shifted = np.zeros_like(u_field)
        u_field_shifted[ii_relief:] = u_field[:-ii_relief]
    return u_field_shifted

# ------- END ------- #
# --- ATMOSPHERES --- #
# ------------------- #
