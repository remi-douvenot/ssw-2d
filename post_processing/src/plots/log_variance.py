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


# -- Load homogeneous fields SSW-2D as refrences --#
e_field_standard_10G_Los10 =  np.load('../post_processing/outputs/E_field_standard_10G_40km_Los100.npy')

# Function that computes the numerical log-amplitude variance
def log_variance(config, E_turbulent):
    # -- Constants-- #
    Cn2 = config.Cn2  # exponent
    c0 = cst.c  # m/s
    Los = config.L0  # Large scale


    # -- Load homogeneous fields SSW-2D as refrences --#

    # sigma2 = np.load('sigma2') #Comme ça à chaque appelle de fonction on appelle le sigma2 moyenné issue des simus précédentes
    n_x = config.N_x
    print(n_x)
    sigma2 = []
    n_z = config.N_z
    n_apo_z = np.int(config.apo_z * n_z)
    for ii_x in range(0, n_x):
        ln_amplitude = []
        print(ii_x)
        for ii_z in range(n_apo_z, n_z - n_apo_z):  # Prise en compte du champ seulement dans la fenêtre d'apodisation
            ln_amplitude.append(np.log(np.abs(E_turbulent[ii_x][ii_z]) / np.abs(E_homogeneous[ii_x][ii_z])))
        sigma2.append(np.var(ln_amplitude))
    # np.save('sigma2')
    return sigma2


#test = log_amplitude_variance_2D(e_field_Homo_10G_Los10_test, e_field_Cn12_Los100_40G_test)

"""
#Computation of the analytic log-amplitude variance 
def logvar_analytic(config,Los,f):
	Kos = 2*np.pi/Los
	k_0 = 2*np.pi*f/3e8
	sigma2 = []
        R = config.N_x*config.x_step
        n_x = config.n_x
	X = np.linspace(100,R,n_x)
	for x in X :
	    #f = lambda K_z,u : Np2dB**2*2*np.pi*k_0**2*x*0.055*Cn2*(K_z**2+ Kos**2)**(-4/3) *0.5 * (1-np.cos(K_z**2*x*u*(1-u)/k_0)) #2D spherical waves
            #F= intg.nquad(f,[[-3, 3],[0,1]],opts = {'limit' : 100 } ) #Warning : choice of interval of integration and number of subdivision
            f = lambda K_z : Np2dB**2*2*np.pi*k_0**2*x*0.055*Cn2*(K_z**2+ Kos**2)**(-4/3) *0.5 * (1-np.sinc(K_z**2*x/k_0)) #Plane waves
            F= intg.quad(f,-1, 1,limit = 1000000)
            sigma2.append(F[0])
            print(x,F[1])
	return sigma2
"""
# 1G
# sigma2_analytic = calc_variance_log(40000,10,100,f[0])




