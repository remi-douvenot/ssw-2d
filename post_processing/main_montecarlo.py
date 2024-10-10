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
# @package main_montecarlo.py
# @author R. Douvenot - V. Darchy
# @date 06/07/2022
# @version V1
#
# @brief computes and plots log-amplitude variance of the electric field along the propagation in turbulent atmosphere
# case
#
# @param[in] config         Class that contains the propagation parameters (see Classes)
#
# @param[out] None          Plots of log-amplitude variance with

# Warning : Before running this file in a turbulent case, one must run it without turbulence in order to save the field
# E_reference
##





# Local Module Imports
import pywt
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import scipy.constants as cst

from src.plots.log_variance import log_variance ,logvar_analytic, phase_variance, diff_phase_variance, phase_var_analytic, compute_log_amplitude_z, compute_log_amplitude_x

# where config is defined
from src.classes_and_files.read_files import read_config



# put relief below the field (upward shift of the field)
def shift_relief(u_field, ii_relief):
    if ii_relief == 0:
        u_field_shifted = u_field
    else:
        u_field_shifted = np.zeros_like(u_field)
        u_field_shifted[ii_relief:] = u_field[:-ii_relief]
    return u_field_shifted


# input: the source information
file_source_config = '../source/inputs/configuration.csv'
# input: main configuration file
file_configuration = '../propagation/outputs/configuration.csv'

# --- Define and fill the config variable that contains all the simulation parameters --- #
config = read_config(file_configuration, file_source_config)
# --------------------------------------------------------------------------------------- #



n_simu = 500 #number of simulations
n_x = config.N_x
n_z = config.N_z
n_apo_z = int(config.apo_z * n_z)
rng = np.random.default_rng()

# --- Initialise tables --- #

# sigma2_table = [np.zeros(n_x)]*n_simu
var_phase_table = [np.zeros(n_x)]*n_simu
# log_ampli_table = [np.zeros(n_x)]*n_simu
# log_ampli_x20_table = [np.zeros(n_z-2*n_apo_z)]*n_simu
# log_ampli_x40_table = [np.zeros(n_z-2*n_apo_z)]*n_simu
log_ampli_x60_table = [np.zeros(n_z-2*n_apo_z)]*n_simu
# log_ampli_x80_table = [np.zeros(n_z-2*n_apo_z)]*n_simu
log_ampli_x95_table = [np.zeros(n_z-2*n_apo_z)]*n_simu
#print(np.shape(log_ampli_x95_table))


# --- Starting simulations --- #
if config.turbulence == 'Y':
    e_reference = np.load('./outputs/E_reference.npy') #Warning : check that this field has been updated before launching turbulent simulations
    for ii_simu in range (n_simu):
        N_screen = config.N_x
        print(N_screen)
        random_index_list = rng.permutation(380)
        print(ii_simu)
        # main program: SSW propagation
        # change directory and launch propagation (depends on the OS)
        if sys.platform == "linux" or sys.platform == "linux2" or sys.platform == "darwin":
            cmd = 'cd ../propagation && python3 ./main_propagation.py'
        else:
            cmd = 'cd ../propagation && python ./main_propagation.py'
        os.system(cmd)
        # --- Load wavelets along x --- #
        wv_total = np.load('../propagation/outputs/wv_total.npy', allow_pickle=True)

        # --- Load relief --- #
        z_relief = np.loadtxt('../terrain/outputs/z_relief.csv', delimiter=',', dtype="float")
        diff_relief = np.diff(z_relief)

        # --- Read the parameter values --- #
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

        wv_ii_x = [[]] * (wv_l + 1)

        # --- Image layer --- #
        ground_type = config.ground
        if ground_type == 'NoGround':  # No ground, no image layer
            n_im = 0
        else:  # ground, therefore an image layer different from 0
            image_layer = config.image_layer  # image_layer in % of the total size n_z
            n_im = np.int(np.round(n_z * image_layer))
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
            u_field_total[ii_x, :] = uu_x[0:] #mettre 0 à la place de n_im si relief
            # add the relief
            if ground_type == 'PEC' or ground_type == 'dielectric':
                # whether ascending or descending relief, the shift is made before or after propagation
                if diff_relief[ii_x] < 0:
                    ii_relief = int(z_relief[ii_x + 1] / z_step)
                else:
                    ii_relief = int(z_relief[ii_x] / z_step)
                u_field_total[ii_x, :] = shift_relief(u_field_total[ii_x, :], ii_relief)
            x_current = x_s + (ii_x + 1) * x_step
            # print('x_current', x_current)

            e_field_total[ii_x, :] = u_field_total[ii_x, :] / np.sqrt(k0 * x_current) * np.exp(-1j * k0 * x_current)
            #e_field_total[ii_x, :] = u_field_total[ii_x, :]
        #field_table[ii_simu]=e_field_total

        print('e_mean')
        # --- Compute log-variance for each simulation  --- #

        # sigma2_table[ii_simu] = log_variance(config,e_field_total,e_reference)

        # var_phase_table[ii_simu] = diff_phase_variance(config,e_field_total, e_reference)

        # log_ampli_table[ii_simu] = compute_log_amplitude_z(config,n_z//2,e_field_total,e_reference)

        # log_ampli_x20_table[ii_simu] = compute_log_amplitude_x(config,config.N_x*20//95,e_field_total,e_reference)
        # log_ampli_x40_table[ii_simu] = compute_log_amplitude_x(config, config.N_x * 40 // 95, e_field_total, e_reference)
        log_ampli_x60_table[ii_simu] = compute_log_amplitude_x(config, config.N_x * 60 // 95, e_field_total, e_reference)
        # log_ampli_x80_table[ii_simu] = compute_log_amplitude_x(config, config.N_x * 80 // 95, e_field_total, e_reference)
        log_ampli_x95_table[ii_simu] = compute_log_amplitude_x(config, config.N_x-1, e_field_total, e_reference)
    # sigma2_mean = np.mean(sigma2_table,axis=0)
    # var_phase_mean = np.mean(var_phase_table,axis=0)
    # log_ampli_mean = np.mean(log_ampli_table,axis = 0)
    # var_phase_std = np.std(var_phase_table, axis=0)
    # log_ampli_std = np.std(log_ampli_table, axis=0)

    #mean log-amplitude profile at every chosen x

    #log_ampli_x20_mean = np.mean(log_ampli_x20_table,axis=0)
    #log_ampli_x40_mean = np.mean(log_ampli_x40_table, axis=0)
    log_ampli_x60_mean = np.mean(log_ampli_x60_table, axis=0)
    #log_ampli_x80_mean = np.mean(log_ampli_x80_table, axis=0)
    log_ampli_x95_mean = np.mean(log_ampli_x95_table, axis=0)

    #corresponding standard deviation

    #log_ampli_x20_std = np.std(log_ampli_x20_table,axis=0)
    #log_ampli_x40_std = np.std(log_ampli_x40_table, axis=0)
    log_ampli_x60_std = np.std(log_ampli_x60_table, axis=0)
    #log_ampli_x80_std = np.std(log_ampli_x80_table, axis=0)
    log_ampli_x95_std = np.std(log_ampli_x95_table, axis=0)

    #sigma2_analytic = logvar_analytic(config)  # Analytic log amplitude variance
    #varphi_analytic = phase_var_analytic(config)  # Analytic log amplitude variance

    ### -- save -- ###

    # Here you save sur mean log-amplitude profile at every chosen x distance as well as the corresponding standard deviation profile

    # np.save('./outputs/extended_LES_logampli_dz1_10GHz_x20_Ns380_Cn023_12', log_ampli_x20_mean)
    # np.save('./outputs/std_extended_LES_logampli_dz1_10GHz_x20_Ns380_Cn023_12', log_ampli_x20_std)
    # np.save('./outputs/extended_LES_logampli_dz1_10GHz_x40_Ns380_Cn023_12', log_ampli_x40_mean)
    # np.save('./outputs/std_extended_LES_logampli_dz1_10GHz_x40_Ns380_Cn023_12', log_ampli_x40_std)
    np.save('/home/victor/simus_propa_source_ARM/ARM_kolmo_logampli_10GHz_x60_dt124s', log_ampli_x60_mean)
    np.save('/home/victor/simus_propa_source_ARM/std_ARM_kolmo_logampli_10GHz_x60_dt124s', log_ampli_x60_std)
    # np.save('./outputs/extended_LES_logampli_dz1_10GHz_x80_Ns380_Cn023_12', log_ampli_x80_mean)
    # np.save('./outputs/std_extended_LES_logampli_dz1_10GHz_x80_Ns380_Cn023_12', log_ampli_x80_std)
    np.save('/home/victor/simus_propa_source_ARM/ARM_kolmo_logampli_10GHz_x95_dt124s', log_ampli_x95_mean)
    np.save('/home/victor/simus_propa_source_ARM/std_ARM_kolmo_logampli_10GHz_x95_dt124s', log_ampli_x95_std)

    # np.save('./outputs/logampli_kolmo_10GHz_x95_Ns95_Cn2_LES_Losx220_zs1500_dz10', log_ampli_x20_mean)
    # np.save('./outputs/std_logampli_kolmo_10GHz_x95_Ns95_Cn2_LES_Losx220_zs1500_dz10', log_ampli_x20_std)
    # np.save('./outputs/logampli_kolmo_10GHz_x95_Ns95_Cn2_LES_Losx220_zs1500_dz10', log_ampli_x40_mean)
    # np.save('./outputs/std_logampli_kolmo_10GHz_x95_Ns95_Cn2_LES_Losx220_zs1500_dz10', log_ampli_x40_std)
    # np.save('./outputs/logampli_kolmo_10GHz_x95_Ns95_Cn2_LES_Losx220_zs1500_dz10', log_ampli_x60_mean)
    # np.save('./outputs/std_logampli_kolmo_10GHz_x95_Ns95_Cn2_LES_Losx220_zs1500_dz10', log_ampli_x60_std)
    # np.save('./outputs/logampli_kolmo_10GHz_x95_Ns95_Cn2_LES_Losx220_zs1500_dz10', log_ampli_x80_mean)
    # np.save('./outputs/std_logampli_kolmo_10GHz_x95_Ns95_Cn2_LES_Losx220_zs1500_dz10', log_ampli_x80_std)
    #np.save('./outputs/logampli_kolmo_LES_10GHz_x95_Ns95_Cn2_020_13_Losx220_zs1500_dz10.npy', log_ampli_x95_mean)
    #np.save('./outputs/std_logampli_kolmo_LES_10GHz_x95_Ns95_Cn2_020_13_Losx220_zs1500_dz10.npy', log_ampli_x95_std)

    ### -- plots -- ###

    x = np.linspace(x_step, config.N_x * config.x_step, config.N_x)
    if config.ground == 'NoGround':
        z = np.linspace(z_apo, n_z *z_step - z_apo, n_z-2*n_apo_z)
    else :
        z = np.linspace(0, n_z * z_step - z_apo, n_z - n_apo_z)
    # #x_analytic = np.linspace(x_step, config.N_x*config.x_step, len(sigma2_analytic))
    # # plt.plot(x/1000, sigma2_mean,'--', label='SSW')
    # # plt.plot(x_analytic/1000, sigma2_analytic,'k', label='Analytic')
    # plt.xlabel('Distance (km)')
    # plt.ylabel('\u03C3$^2$ (dB$^2$)')
    # #plt.xlim(7.5,40)
    # #plt.ylim(0, 0.12)
    # plt.title('Log-amplitude variance at '+str(config.freq/1e9)+' GHz')
    # plt.grid()
    # plt.legend()
    # plt.show()

    # plt.plot(x / 1000, log_ampli_mean, '--', label='SSW')
    # # plt.plot(x_analytic/1000, sigma2_analytic, label='Analytic')
    # plt.xlabel('Distance (km)')
    # plt.ylabel('$\chi$ (dB)')
    # # plt.xlim(7.5,40)
    # # plt.ylim(0, 0.12)
    # plt.title('Log-amplitude at ' + str(config.freq / 1e9) + ' GHz ; z = 1500 m')
    # plt.grid()
    # plt.legend()
    # plt.show()


    #plt.plot(x/1000, np.sqrt(var_phase_mean)*180/np.pi,'-', label='SSW')
    #plt.plot(x / 1000, var_phase_mean+var_phase_std, '--')
    #plt.plot(x / 1000, var_phase_mean - var_phase_std, '--')
    #plt.plot(x_analytic / 1000, varphi_analytic, label='Analytic')
    #plt.xlabel('Distance (km)')
    #plt.ylabel('phase variance')
    #plt.xlim(7.5,40)
    #plt.ylim(0, 0.12)
    #plt.title('Phase variance at '+str(config.freq/1e9)+' GHz ; Los = 400 m ; Cn2 = 1e-12')
    #plt.grid()
    #plt.legend()
    #plt.show()

### Plots log-ampli profiles

    # plt.plot(log_ampli_x20_mean, z , 'k-', label='x = 20 km')
    # plt.plot(log_ampli_x20_mean+log_ampli_x20_std, z, 'r:')
    # plt.plot(log_ampli_x20_mean - log_ampli_x20_std, z, 'r:')
    # plt.xlabel('$\chi$ (dB)')
    # plt.ylabel('z(m)')
    # # plt.xlim(7.5,40)
    # # plt.ylim(0, 0.12)
    # plt.title('Mean log-amplitude at ' + str(config.freq / 1e9) + ' GHz ; x = 20 km ; 500 runs ')
    # plt.grid()
    # plt.legend()
    # plt.show()
    #
    # plt.plot(log_ampli_x40_mean, z , 'k-', label='x = 40 km')
    # plt.plot(log_ampli_x40_mean+log_ampli_x40_std, z, 'r:')
    # plt.plot(log_ampli_x40_mean - log_ampli_x40_std, z, 'r:')
    # plt.xlabel('$\chi$ (dB)')
    # plt.ylabel('z(m)')
    # # plt.xlim(7.5,40)
    # # plt.ylim(0, 0.12)
    # plt.title('Mean log-amplitude at ' + str(config.freq / 1e9) + ' GHz ; x = 40 km ; 500 runs ')
    # plt.grid()
    # plt.legend()
    # plt.show()
    #
    # plt.plot(log_ampli_x60_mean, z , 'k-', label='x = 60 km')
    # plt.plot(log_ampli_x60_mean+log_ampli_x60_std, z, 'r:')
    # plt.plot(log_ampli_x60_mean - log_ampli_x60_std, z, 'r:')
    # plt.xlabel('$\chi$ (dB)')
    # plt.ylabel('z(m)')
    # # plt.xlim(7.5,40)
    # # plt.ylim(0, 0.12)
    # plt.title('Mean log-amplitude at ' + str(config.freq / 1e9) + ' GHz ; x = 60 km ; 500 runs ')
    # plt.grid()
    # plt.legend()
    # plt.show()
    #
    # plt.plot(log_ampli_x80_mean, z , 'k-', label='x = 80 km')
    # plt.plot(log_ampli_x80_mean+log_ampli_x80_std, z, 'r:')
    # plt.plot(log_ampli_x80_mean - log_ampli_x80_std, z, 'r:')
    # plt.xlabel('$\chi$ (dB)')
    # plt.ylabel('z(m)')
    # # plt.xlim(7.5,40)
    # # plt.ylim(0, 0.12)
    # plt.title('Mean log-amplitude at ' + str(config.freq / 1e9) + ' GHz ; x = 80 km ; 500 runs ')
    # plt.grid()
    # plt.legend()
    # plt.show()

    plt.plot(log_ampli_x95_mean, z , 'k-', label='x = 95 km')
    plt.plot(log_ampli_x95_mean+log_ampli_x95_std, z, 'r:')
    plt.plot(log_ampli_x95_mean - log_ampli_x95_std, z, 'r:')
    plt.xlabel('$\chi$ (dB)')
    plt.ylabel('z (m)')
    # plt.xlim(7.5,40)
    # plt.ylim(0, 0.12)
    #plt.title('Mean log-amplitude at ' + str(config.freq / 1e9) + ' GHz ; x = 95 km ; 500 runs ')
    plt.title('$\chi$ profile at 95 km')
    plt.grid()
    plt.legend()
    plt.show()



else : #save non turbulent field
    
    # main program: SSW propagation
    # change directory and launch propagation (depends on the OS)
    if sys.platform == "linux" or sys.platform == "linux2" or sys.platform == "darwin":
        cmd = 'cd ../propagation && python3 ./main_propagation.py'
    else:
        cmd = 'cd ../propagation && python ./main_propagation.py'
    os.system(cmd)
    # --- Load wavelets along x --- #
    wv_total = np.load('../propagation/outputs/wv_total.npy', allow_pickle=True)

    # --- Load relief --- #
    z_relief = np.loadtxt('../terrain/outputs/z_relief.csv', delimiter=',', dtype="float")
    diff_relief = np.diff(z_relief)

    # --- Read the parameter values --- #
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
    n_apo_z = int(np.round(n_z * config.apo_z))

    # --- Initialise field --- #
    u_field_total = np.zeros((n_x, n_z), dtype='complex')
    e_field_total = np.zeros((n_x, n_z), dtype='complex')

    wv_ii_x = [[]] * (wv_l + 1)

    # --- Image layer --- #
    ground_type = config.ground
    if ground_type == 'NoGround':  # No ground, no image layer
        n_im = 0
    else:  # ground, therefore an image layer different from 0
        image_layer = config.image_layer  # image_layer in % of the total size n_z
        n_im = int(round(n_z * image_layer))
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
        print(np.shape(uu_x))
        print(np.shape(u_field_total))
        # remove image field
        u_field_total[ii_x, :] = uu_x[0:] #mettre 0 à la place de n_im si relief
        # add the relief
        if ground_type == 'PEC' or ground_type == 'dielectric':
            # whether ascending or descending relief, the shift is made before or after propagation
            if diff_relief[ii_x] < 0:
                ii_relief = int(z_relief[ii_x + 1] / z_step)
            else:
                ii_relief = int(z_relief[ii_x] / z_step)
            u_field_total[ii_x, :] = shift_relief(u_field_total[ii_x, :], ii_relief)
        x_current = x_s + (ii_x + 1) * x_step
        # print('x_current', x_current)

        e_field_total[ii_x, :] = u_field_total[ii_x, :] / np.sqrt(k0 * x_current) * np.exp(-1j * k0 * x_current)
        #e_field_total[ii_x, :] = u_field_total[ii_x, :]

    #var_phase_table = phase_variance(config,e_field_total)
    #sigma2_mean = np.mean(sigma2_table,axis=0)
    #var_phase_mean = var_phase_table
    #np.save('./outputs/extended_LES_logvar_dz50_1GHz_Ns380_Cn2_023',sigma2_mean)
    #np.save('./outputs/var_phase_1GHz', var_phase_mean)
    #sigma2_analytic = logvar_analytic(config)  # Analytic log amplitude variance
    #x = np.linspace(x_step, config.N_x * config.x_step, config.N_x)
    #x_analytic = np.linspace(x_step, config.N_x*config.x_step, len(sigma2_analytic))
    #plt.plot(x/1000, sigma2_mean,'--', label='SSW')
    #plt.plot(x_analytic/1000, sigma2_analytic, label='Analytic')
    #plt.xlabel('Distance (km)')
    #plt.ylabel('\u03C3$^2$ (dB$^2$)')
    #plt.xlim(7.5,40)
    #plt.ylim(0, 0.12)
    #plt.title('Log-amplitude variance at '+str(config.freq/1e9)+' GHz ; LES BOMEX LES atmosphere : dz = 50 m')
    #plt.grid()
    #plt.legend()
    #plt.show()

    # plt.plot(x[200:]/1000, var_phase_mean[200:],'--', label='SSW')
    # #plt.plot(x_analytic/1000, sigma2_analytic, label='Analytic')
    # plt.xlabel('Distance (km)')
    # plt.ylabel('phase variance')
    # #plt.xlim(7.5,40)
    # #plt.ylim(0, 0.12)
    # plt.title('Phase variance at '+str(config.freq/1e9)+' GHz ; LES BOMEX LES atmosphere : dz = 50 m')
    # plt.grid()
    # plt.legend()
    # plt.show()


    np.save('./outputs/E_reference', e_field_total)
    # print('Reference field is saved for f =',config.freq*1e-6,'MHz and L0 =',config.L0, 'm')
    print('Reference field is saved for f =', config.freq * 1e-6, 'MHz')
    print('Launch turbulent simulation')
