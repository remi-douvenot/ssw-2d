##
# @brief Function that generates the vertical profile of n or m index
# @author R. Douvenot
# @package genere_n_profile.py
# @date 08/09/21
# @version Work in progress
#
# @details Function that generates the vertical profile of n or m index. Available atmospheres: homogeneous, standard,
# bilinear, trilinear, evaporation duct, double duct, loaded from a file.
# n_refraction = genere_n_profile(config)
#
# @params[in] config : class with the parameters
# @params[out] n_profile : vertical profile of n or m (modified) refractive index
##

import numpy as np
import scipy.constants as cst
import matplotlib.pyplot as plt
import scipy.signal

def lissage(signal_brut,L):
    res = np.copy(signal_brut) # duplication des valeurs
    for i in range (1,len(signal_brut)-1): # toutes les valeurs sauf la première et la dernière
        L_g = min(i,L) # nombre de valeurs disponibles à gauche
        L_d = min(len(signal_brut)-i-1,L) # nombre de valeurs disponibles à droite
        Li=min(L_g,L_d)
        res[i]=np.sum(signal_brut[i-Li:i+Li+1])/(2*Li+1)
    return res

def low_pass_filter(input_array, factor):
    # Créer un noyau passe-bas
    kernel_size = factor * 2 + 1
    kernel = np.ones(kernel_size) / kernel_size

    # Appliquer la convolution avec le noyau
    filtered_array = np.convolve(input_array, kernel, mode='same')

    # Sous-échantillonner en prenant un point sur 'factor' points
    filtered_array = filtered_array[::factor]

    return filtered_array

def generate_n_profile(config):
    # choice of the atmospheric profile type
    if config.atmosphere == 'Homogeneous':  # constant n (vacuum)
        n_refractive_index = np.ones(config.N_z)
    elif config.atmosphere == 'Linear':  # constant slope
        n_refractive_index = linear_atmosphere(config.N_z, config.z_step, config.c0)
    elif config.atmosphere == 'Evaporation':  # log then constant slope
        n_refractive_index = evaporation_duct(config.N_z, config.z_step, config.c0, config.delta)
    elif config.atmosphere == 'Bilinear':  # bilinear profile = trilinear with zb = 0
        n_refractive_index = trilinear_profile(config.N_z, config.z_step, config.c0, 0.0, config.c2, config.zt)
    elif config.atmosphere == 'Trilinear':  # trilinear
        n_refractive_index = trilinear_profile(config.N_z, config.z_step, config.c0, config.zb, config.c2, config.zt)
    elif config.atmosphere == 'Double duct':  # log then trilinear
        n_refractive_index = double_duct(config.N_z, config.z_step, config.c0, config.delta, config.zb, config.c2, config.zt)
    elif config.atmosphere == 'file':  # complex profile from file
        n_refractive_index = read_file_profile(config.N_z, config.z_step, config.atm_filename)
    else:
        raise ValueError(['Wrong atmospheric type!'])

    return n_refractive_index

def genere_phi_turbulent(config):
    if config.turbulence == 'Y':
        ##-- MPS --##
        z = np.linspace(0, config.N_z * config.z_step, config.N_z)
        Cn2_z_profile = np.load('./src/atmosphere/Cn2_y_dxyz5.npy')
        z_LES = np.linspace(0, 3000, np.size(Cn2_z_profile))
        k0 = 2 * np.pi * config.freq / cst.c
        phi_turbulent = turbulent(config.N_z, config.z_step, config.x_step,config.L0,0,config.freq) #0 for Cn2 exponent if the Cn2 profile is applied after generation (LES Kolmo or BK-Model). If so, the profile is applied in the ssw2d function
        phi_turbulent*=np.interp(z,z_LES,np.sqrt(Cn2_z_profile))
    return phi_turbulent

def genere_phi_turbulent_LES(config,phi_LES):
    if config.turbulence == 'Y':
        ##-- LES --##
        #phi_LES = scipy.signal.decimate(phi_LES,2)
        #phi_LES = low_pass_filter(phi_LES,2)
        z_max = config.N_z*config.z_step
        k0 = 2 * np.pi * config.freq / cst.c
        N_z_LES = np.size(phi_LES)
        z_LES = np.linspace(0,3000,N_z_LES) #3000 if BOMEX, 4400 if ARM
        Delta_LES = config.N_z*config.z_step/N_z_LES
        #inhomo_vector = np.zeros(config.N_z)
        #inhomo_vector[:int(2000//config.z_step)] = np.ones(np.size(inhomo_vector[:int(2000//config.z_step)]))
        z = np.linspace(0, config.N_z * config.z_step, config.N_z)
        phi_LES_tronc = phi_LES[:int((config.N_z*config.z_step)*N_z_LES/3000)] #3000 if BOMEX, 4400 if ARM
        z_LES_tronc = np.linspace(0,config.N_z*config.z_step,np.size(phi_LES_tronc))
        Cn2_profile = np.interp(z,z_LES_tronc,np.load('./src/atmosphere/Cn2_x_dxyz5.npy')[:int((config.N_z*config.z_step)*N_z_LES/3000)]) #choose the needed Cn2 profile; 3000 if BOMEX, 4400 if ARM
        #Cn2_profile = np.interp(z, z_LES_tronc, np.load('./src/atmosphere/Cn2_x_arm_22h_1h_dxyz10.npy')[:int((config.N_z * config.z_step) * N_z_LES / 4395)])
        phi_resolved = k0*np.interp(z, z_LES_tronc, phi_LES_tronc)  # interpolation between the LES grid and propagation grid
        phi_subgrid = k0 * np.sqrt(Cn2_profile)* turbulent_subgrid(config.N_z, config.z_step, N_z_LES, Delta_LES, config.x_step, 0) #LES-Kolmogorov
        #phi_subgrid = k0 *  turbulent_subgrid(config.N_z, config.z_step, N_z_LES, Delta_LES, config.x_step, config.Cn2) #Homogeneous extension
        #phi_subgrid = phi_subgrid * inhomo_vector
        phi_turbulent = phi_subgrid + phi_resolved
        #phi_turbulent = phi_resolved


    return phi_turbulent

# standard atmosphere
def linear_atmosphere(n_z, z_step, c0):
    # vector of vertical positions
    z_vect = np.linspace(0, z_step*n_z, n_z, endpoint=False)
    # refractivity
    # mean M0 is 330 at the ground level
    n_refractivity = 330 + z_vect*c0
    # refractive index
    n_refractive_index = 1 + n_refractivity*1e-6
    return n_refractive_index


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
    # in the duct: refractivity following Paulus-Jeske model
    n_refractivity[indices_z_inf] = 330 + 0.125*(z_vect[indices_z_inf] -
                                    delta*np.log((z_vect[indices_z_inf]+z0) / z0))
    # above the duct: standard atm
    n_refractivity[indices_z_sup] = 330 + z_vect[indices_z_sup]*c0
    # ensuring continuity
    n_refractivity[indices_z_inf] = n_refractivity[indices_z_inf] - n_refractivity[indices_z_inf[0][-1]] \
                                  + n_refractivity[indices_z_sup[0][0]] - c0*z_step
    # refractive index
    n_refractive_index = 1 + n_refractivity * 1e-6
    return n_refractive_index
# This file is part of SSW-2D.
# SSW-2D is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# SSW-2D is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
# of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Foobar. If not, see
# <https://www.gnu.org/licenses/>.

# # bilinear profile
# def bilinear_profile(n_z, z_step, c0, c2, zt):
#     raise ValueError(['bilinear_profile duct must be validated'])
#     # vector of vertical positions
#     z_vect = np.linspace(0, z_step*n_z, n_z, endpoint=False)
#     n_refractivity = np.zeros_like(z_vect)
#     # --- refractivity --- #
#     # mean M0 is 330 at the ground level
#     # in the duct
#     n_refractivity[z_vect <= zt] = 330 + z_vect[z_vect <= zt]*c2
#     # above the duct
#     n_refractivity[z_vect > zt] = 330 + zt*c2 + (z_vect[z_vect > zt]-zt) * c0
#     # --- refractive index --- #
#     n_refractive_index = 1 + n_refractivity*1e-6
#     return n_refractive_index


# trilinear profile
def trilinear_profile(n_z, z_step, c0, zb, c2, zt):
    # vector of vertical positions
    z_vect = np.linspace(0, z_step * n_z, n_z, endpoint=False)
    n_refractivity = np.zeros_like(z_vect)
    # z0
    z0 = 1.5e-4
    # print(indices_z_inf)
    # below the duct
    indices_zb = np.where(z_vect <= zb)  # below the duct
    n_refractivity[indices_zb] = 330 + z_vect[indices_zb] * c0
    # in the duct
    indices_zt = np.where(z_vect > zb)
    n_refractivity[indices_zt] = 330 + zb * c0 + (z_vect[indices_zt] - zb) * c2
    # above the duct
    indices_above = np.where(z_vect > zt + zb)  # above the duct
    n_refractivity[indices_above] = 330 + zb * c0 + zt * c2 + (z_vect[indices_above] - zb - zt) * c0
    # --- refractive index --- #
    n_refractive_index = 1 + n_refractivity*1e-6
    return n_refractive_index


# double duct
def double_duct(n_z, z_step, c0, delta, zb, c2, zt):
    raise ValueError(['double duct not yet coded'])
    n_refractive_index = 1
    return n_refractive_index


# read profile from a text file
def read_file_profile(n_z, z_step, atm_filename):
    raise ValueError(['read_file_profile not yet coded'])
    n_refractive_index = 1
    return n_refractive_index

#--- kolmogorov turbulence ---#

# function that generates a turbulent phase screen based on a Von-Karman Kolmogorov spectrum.
# Warning : S_Phi2D is the classical VKK spectrum multiplied by N_z in order to be consistent with the general
# convention of orthonormal normalization used in all the code. (see future REF)

def turbulent(n_z, z_step, x_step, Los, Cn2_exponent,f):
    k0 = 2*np.pi*f / cst.c
    Kos = 2*np.pi/Los
    z = np.linspace(0,n_z*z_step,n_z)
    # Compute spectral discretization
    #q_z = np.linspace(0,n_z-1, num=n_z, endpoint = True)
    q_z = np.linspace(-n_z/2, n_z/2 - 1, num=n_z, endpoint=True)
    k_z = (2/z_step) * np.sin(np.pi*q_z/(n_z)) #version DSSF
    #k_z = (1*np.pi)/(n_z*z_step)*q_z #version SSF
    # Define Von Karman Kolmogorov (VKK) spectrum
    S_Phi2D = (2*np.pi)* k0**2*x_step*0.055*10**Cn2_exponent*(k_z**2+Kos**2)**(-4/3) #normalization of VKK spectrum by S*n_z

    # Generate random gaussian white noise filtred by a VKK spectrum

    #a = np.random.normal(0,np.sqrt((2*np.pi)/(n_z*z_step))*np.sqrt(S_Phi2D),n_z) #SSF
    #b = np.random.normal(0,np.sqrt((2*np.pi)/(n_z*z_step))*np.sqrt(S_Phi2D),n_z) #SSF

    a = np.random.normal(0,np.sqrt((2/z_step)*np.sin(np.pi/n_z))*np.sqrt(S_Phi2D),n_z) #DSSF
    b = np.random.normal(0,np.sqrt((2/z_step)*np.sin(np.pi/n_z))*np.sqrt(S_Phi2D),n_z) #DSSF
    gauss = (a + 1j*b)
    print('energy gauss', np.sum(np.abs(gauss)**2))
    G = np.fft.fftshift(gauss) #DSSF
    # --- turbulent phase screen --- #
    Phi=n_z*(np.fft.ifft(G).real)
    print('energy phi',np.sum(np.abs(Phi)**2)/n_z)
    return Phi

def turbulent_subgrid(n_z, z_step, n_z_LES, Delta_LES, x_step, Cn2_exponent):
    ### Enable to introduce
    # Compute spectral discretization
    tronc_factor = n_z*z_step/(n_z_LES*Delta_LES)
    print('z_max',n_z * z_step)
    print('z_les',n_z_LES*Delta_LES)
    print('tronc factor =',tronc_factor)
    q_z_neg = np.linspace(-n_z/2, -int(n_z*z_step//(2*Delta_LES)), num=int(n_z/2 - tronc_factor*n_z_LES/2), endpoint=True) #negative part of the spectrum
    q_z_pos = np.linspace(int(n_z * z_step // (2 * Delta_LES)), n_z /2 -1, num=int(n_z/2 - tronc_factor*n_z_LES/2), endpoint=True) #positive part of the spectrum

    k_z_neg = (2*np.pi)/(n_z*z_step)*q_z_neg
    k_z_pos = (2*np.pi) / (n_z * z_step) * q_z_pos
    # Define Von Karman Kolmogorov (VKK) spectrum
    print(int(n_z*z_step//Delta_LES*tronc_factor))
    n_delta = int(n_z*z_step//Delta_LES*tronc_factor)

    zeros = np.zeros(int(n_z*z_step//Delta_LES)) #Lowest part of the spectrum: do not need to be generated (already by LES)
    print('zeros =',np.size(zeros))
    print('kz neg = ',np.size(k_z_neg))
    print('kz neg = ',np.size(k_z_pos))
    S_Phi2D = np.concatenate(((2*np.pi)* x_step*0.055*0.13*10**(Cn2_exponent)*(k_z_neg**2)**(-4/3),zeros,(2*np.pi)* x_step*0.055*10**(Cn2_exponent)*(k_z_pos**2)**(-4/3))) #Concatenation => complete spectrum; 0.13 for Cn2 = 0.13x10^-12
    print(np.size(S_Phi2D),n_z)
    # Generate random gaussian white noise filtred by a VKK spectrum

    a = np.random.normal(0,np.sqrt((2/z_step)*np.sin(np.pi/n_z))*np.sqrt(S_Phi2D),n_z) #DSSF
    b = np.random.normal(0,np.sqrt((2/z_step)*np.sin(np.pi/n_z))*np.sqrt(S_Phi2D),n_z) #DSSF
    gauss = (a + 1j*b)
    print('energy gauss', np.sum(np.abs(gauss)**2))
    #fft shift to remove symmetry
    #G = gauss #DSSF
    G = np.fft.fftshift(gauss) #SSF

    # --- turbulent phase screen --- #
    Phi=n_z*(np.fft.ifft(G).real)
    print('energy phi',np.sum(np.abs(Phi)**2)/n_z)

    return Phi