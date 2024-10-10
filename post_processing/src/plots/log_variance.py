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
import matplotlib.pyplot as plt
import math as m
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.colors
from scipy import integrate as intg

##
# @package logvar_MPS
# @author R. Douvenot - V. Darchy
# @date 06/07/2022
# @version V1
#
# @brief computes and plots log-amplitude variance of the electric field along the propagation in turbulent atmosphere case
#
# @param[in] config         Class that contains the propagation parameters (see Classes)
#
# @param[out] None          Plots are displayed and saved in the "outputs" directory

# Usable only if ground == none
##





#Computation of the numerical log-amplitude variance
def log_variance(config, E_turbulent,E_reference):
    # --- set parameters --#
    n_x = config.N_x
    sigma2 = np.zeros(n_x)
    n_z = config.N_z
    n_apo_z = np.int(config.apo_z * n_z)
    x_max = config.N_x * config.x_step
    print(n_apo_z)
    # --- compute vertical log-variance at each range step --#
    Np2dB = 8.686
    for ii_x in range(0, n_x):
        ln_amplitude = np.zeros(n_z-2*n_apo_z)
        #print(ii_x)
        for ii_z in range(n_apo_z, n_z - n_apo_z): #Field is true only in apo window
            #ln_amplitude[ii_z-n_apo_z]= Np2dB*np.sqrt(x_max/(x_max-config.x_s))*np.log(np.abs(E_turbulent[ii_x][ii_z])/np.abs(E_reference[ii_x][ii_z]))
            ln_amplitude[ii_z - n_apo_z] = Np2dB * np.log(np.abs(E_turbulent[ii_x][ii_z]) / np.abs(E_reference[ii_x][ii_z]))
        sigma2[ii_x]= np.var(ln_amplitude)
    # np.save('sigma2')
    print('max champ',np.max(20*np.log10(np.abs(E_turbulent[-1]))),np.max(20*np.log10(np.abs(E_reference[-1]))))
    return sigma2



#Computation of the analytic log-amplitude variance 
def logvar_analytic(config):
    #--- set parameters --#
    Kos = 2*np.pi/config.L0
    k_0 = 2*np.pi*config.freq/3e8
    n_x = config.N_x
    R = n_x*config.x_step  #total range
    N_ite = 10
    x_s = - config.x_s
    X = np.linspace(100,R,N_ite)
    sigma2 = np.zeros(N_ite)
    Np2dB = 8.686
    # --- compute vertical log-variance at each range step --#
    for ii_x in range(N_ite) :
        x = X[ii_x]
        f = lambda k_z,u : Np2dB**2*2*np.pi*k_0**2*x*0.055*10**(config.Cn2)*(k_z**2+ Kos**2)**(-4/3) *0.5 * (1-np.cos(k_z**2*x*u*(1-u)/k_0)) #2D spherical waves
        F= intg.nquad(f,[[-3, 3],[0,1]],opts = {'limit' : 100000000 } ) #Warning : choice of interval of integration and number of subdivision
        #f = lambda k_z : Np2dB**2*2*np.pi*k_0**2*x*0.055*10**(config.Cn2)*(k_z**2+ Kos**2)**(-4/3) *0.5 * (1-np.sinc(k_z**2*x/k_0)) #Plane waves
        #F= intg.quad(f,-3, 3,limit = 100000000)
        sigma2[ii_x] = F[0]
        print(x,F[1])
    return sigma2

#Computation of the analytic phase variance
def phase_var_analytic(config):
    #--- set parameters --#
    Kos = 2*np.pi/config.L0
    k_0 = 2*np.pi*config.freq/3e8
    n_x = config.N_x
    R = n_x*config.x_step  #total range
    N_ite = 5
    x_s = - config.x_s
    X = np.linspace(100,R,N_ite)
    varphi = np.zeros(N_ite)
    # --- compute vertical log-variance at each range step --#
    for ii_x in range(N_ite) :
        x = X[ii_x]
        f = lambda k_z,u : 2*np.pi*k_0**2*x*0.055*10**(config.Cn2)*(k_z**2+ Kos**2)**(-4/3) *k_z*(np.cos(k_z**2*x*u*(x-u)/(2*k_0)))**2 #2D spherical waves
        F= intg.nquad(f,[[0, 3],[0,1]],opts = {'limit' : 1000000 } ) #Warning : choice of interval of integration and number of subdivision
        print(F[0])
        #f = lambda k_z : Np2dB**2*2*np.pi*k_0**2*x*0.055*10**(config.Cn2)*(k_z**2+ Kos**2)**(-4/3) *0.5 * (1-np.sinc(k_z**2*x/k_0)) #Plane waves
        #F= intg.quad(f,-3, 3,limit = 100000000)
        varphi[ii_x] = F[0]
        print(x,F[1])
    return varphi


def phase_variance(config,E_turbulent):
    # --- set parameters --#
    n_x = config.N_x
    var_phi = np.zeros(n_x)
    n_z = config.N_z
    n_apo_z = np.int(config.apo_z * n_z)
    print(n_apo_z)
    # --- compute vertical phase variance at each range step --#
    z = np.linspace(n_apo_z*config.z_step, (n_z-n_apo_z)*config.z_step, n_z - 2*n_apo_z)
    for ii_x in range(0, n_x):
        phi = np.unwrap(np.angle(E_turbulent[ii_x]))
        # if ii_x == 20 or ii_x == 100 or ii_x == 200 or ii_x == 370 :
        #    plt.plot(np.unwrap(phi[n_apo_z:n_z-n_apo_z])*180/np.pi,z)
        #    plt.show()
        #    plt.xlabel('phase (rad)')
        #    plt.ylabel('z (m)')
        # print('size =',np.size(phi))
        # print(phi)
        var_phi[ii_x] = np.var(phi[n_apo_z:n_z-n_apo_z])
    return  var_phi

def diff_phase_variance(config,E_turbulent, E_reference):
    # --- set parameters --#
    n_x = config.N_x
    var_phi = np.zeros(n_x)
    n_z = config.N_z
    n_apo_z = np.int(config.apo_z * n_z)
    print(n_apo_z)
    # --- compute vertical phase variance at each range step --#
    z = np.linspace(n_apo_z*config.z_step, (n_z-n_apo_z)*config.z_step, n_z - 2*n_apo_z)
    for ii_x in range(0, n_x):
        #phi = np.angle(E_turbulent[ii_x] / E_reference[ii_x])
        phi = np.unwrap(np.angle(E_turbulent[ii_x]))-np.unwrap(np.angle(E_reference[ii_x]))
        #print(phi[n_apo_z:n_z-n_apo_z]*180/np.pi)

        # if ii_x == 75 or ii_x == 100 or ii_x == 150 or ii_x == 200 or ii_x == 370 :
        #     plt.plot(np.unwrap(phi[n_apo_z:n_z-n_apo_z]*180/np.pi),z, label = 'unwrap')
        #     print(np.mean(phi[n_apo_z:n_z-n_apo_z]*180/np.pi))
        #     #plt.plot(phi[n_apo_z:n_z - n_apo_z]*180/np.pi, z, label = 'classic')
        #     plt.xlabel('phase shift (°)')
        #     plt.ylabel('z (m)')
        #     plt.title('Phase shift at x='+str(config.x_step*ii_x/1000)+' km for f ='+str(config.freq/1e9)+ ' GHz')
        #     plt.legend()
        #     plt.show()

        #print('size =',np.size(phi))
        #print(phi)
        var_phi[ii_x] = np.var(phi[n_apo_z :n_z - n_apo_z])
        #var_phi[ii_x] = np.var(phi[n_apo_z+150*n_z//500:n_z-(n_apo_z+150*n_z//500)])
    return  var_phi


def compute_log_amplitude_z(config,index_altitude,E_turbulent, E_reference):
    n_x = config.N_x
    Np2dB = 8.686
    x_max = config.N_x * config.x_step
    # ln_ampli_vect =  Np2dB*np.sqrt(x_max/(x_max-config.x_s))*np.log(np.abs(E_turbulent[:,index_altitude])/np.abs(E_reference[:,index_altitude]))
    ln_ampli_vect = Np2dB *  np.log(np.abs(E_turbulent[:, index_altitude]) / np.abs(E_reference[:, index_altitude]))
    return ln_ampli_vect


def compute_log_amplitude_x(config,index_range, E_turbulent, E_reference):
    n_z = config.N_z
    x_max = config.N_x * config.x_step
    n_apo_z = np.int(config.apo_z * n_z)
    Np2dB = 8.686
    if config.ground == 'NoGround':
    # ln_ampli_vect = Np2dB*np.sqrt(x_max/(x_max-config.x_s))*np.log(np.abs(E_turbulent[index_range,n_apo_z:n_z-n_apo_z])/np.abs(E_reference[index_range,n_apo_z:n_z-n_apo_z]))
        ln_ampli_vect = Np2dB*np.log(np.abs(E_turbulent[index_range,n_apo_z:n_z-n_apo_z])/np.abs(E_reference[index_range,n_apo_z:n_z-n_apo_z]))
    else :
        ln_ampli_vect = Np2dB * np.log(np.abs(E_turbulent[index_range,:n_z - n_apo_z]) / np.abs(E_reference[index_range,:n_z - n_apo_z]))
    return ln_ampli_vect