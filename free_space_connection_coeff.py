import pywt
import numpy as np
from source.src.complex_source_point import complex_source_point
import csv
import scipy.constants as cst
import matplotlib.pyplot as plt
from propagation.src.propagation.wgm_one_step import wgm_one_step, galerkin_matrices
from propagation.src.propagation.apodisation import apply_apodisation, apodisation_window
from propagation.src.atmosphere.genere_n_profile import generate_n_profile
from propagation.src.classes_and_files.read_files import read_config
import sys

# ----------------------------- #
# --- Extract configuration --- #
# ----------------------------- #
file_source = 'source/inputs/configuration.csv'


class ConfigSource:
    def __init__(self):
        self.n_z = 0
        self.z_step = 0
        # position of the CSP along x (real <0)
        self.x_s = 0
        # position of the CSP along z (real)
        self.z_s = 0
        # width of the complex source point (real)
        self.W0 = 0
        # radiated power in W (real)
        self.P_Tx = 0
        # max gain in dBi (real)
        self.G_Tx = 0
        # wavenumber in m
        self.k0 = 0
        # altitude of the source in m
        self.z_s = 0
        # width of the aperture W0 (for CSP)
        self.W0 = 0


f_source = open(file_source, newline='')
source_tmp = csv.reader(f_source)
for row in source_tmp:
    if row[0] == 'N_z':
        ConfigSource.n_z = np.int64(row[1])  # change
    elif row[0] == 'z_step':
        ConfigSource.z_step = np.float64(row[1])
    elif row[0] == 'x_s':  # position along x --> must be <0
        ConfigSource.x_s = np.float64(row[1])
    elif row[0] == 'type':
        source_type = row[1]
    elif row[0] == 'frequency':
        freq = np.float64(row[1]) * 1e6
    elif row[0] == 'P_Tx':  # radiated power
        ConfigSource.P_Tx = np.float64(row[1])
    elif row[0] == 'G_Tx':  # antenna max gain (dBi)
        ConfigSource.G_Tx = np.float64(row[1])
    elif row[0] == 'W0':  # waist of the CSP
        ConfigSource.W0 = np.float64(row[1])
    elif row[0] == 'z_s':  # position along z
        ConfigSource.z_s = np.float64(row[1])
    elif row[0] == 'Property':  # first line
        pass
    else:
        raise ValueError(['Input file of the geometry is not valid. Input "' + row[0] + '" not valid'])

ConfigSource.k0 = 2 * cst.pi * freq / cst.c

file_config = 'propagation/inputs/configuration.csv'
propaConfig = read_config(file_config)

# ------------ END ------------ #
# --- Extract configuration --- #
# ----------------------------- #


# --------------------- #
# --- INITIAL FIELD --- #
# --------------------- #

# compute E field
e_field = complex_source_point(ConfigSource)  # initial field

# --- Calculate u_0 from E_init (normalised in infinity norm to have max(|u_0|) = 1) --- #
u_0 = e_field * np.sqrt(ConfigSource.k0 * (-ConfigSource.x_s)) * np.exp(-1j * ConfigSource.k0 * (-ConfigSource.x_s))
# u_0 = e_field * np.exp(1j * ConfigSource.k0 * (-ConfigSource.x_s))
u_infty = np.max(np.abs(u_0))  # norm infinity of the initial field
u_0 /= u_infty  # put max at 1 to avoid numerical errors

# -------- END -------- #
# --- INITIAL FIELD --- #
# --------------------- #

# --- Extended domain --- #
wav = pywt.Wavelet(propaConfig.wv_family)
genus = wav.number
ext_dom = np.zeros(genus-1)
sup_len = propaConfig.N_z + 2 * (genus - 1)  # length of support of the extended domain
u_x = np.concatenate((ext_dom, u_0, ext_dom))

e_total = np.zeros((propaConfig.N_z, propaConfig.N_x), dtype='complex')

# Apodisation window
n_apo_z = np.int64(propaConfig.apo_z * propaConfig.N_z)
apo_window_z = apodisation_window(propaConfig.apo_window, n_apo_z)
apo_window_z = np.concatenate((apo_window_z, np.zeros(genus-1)))

# Generate n profile
n_refractive_index = generate_n_profile(propaConfig)
n_refractive_index_up = n_refractive_index[-genus - 1:-1]
n_refractive_index_down = n_refractive_index[0:genus - 2]
n_refractive_index = np.concatenate((n_refractive_index_down, n_refractive_index, n_refractive_index_up))

# Propagation
propagation_matrix = galerkin_matrices(propaConfig, sup_len, n_refractive_index)

for ii_x in np.arange(0, propaConfig.N_x):
    u_x = apply_apodisation(u_x, apo_window_z, propaConfig)
    u_x_dx = wgm_one_step(u_x, propagation_matrix)
    e_x_dx = u_x_dx * u_infty / np.sqrt(ConfigSource.k0 * (-ConfigSource.x_s + ii_x * propaConfig.x_step)) * np.exp(
        1j * ConfigSource.k0 * (-ConfigSource.x_s + ii_x * propaConfig.x_step))
    e_total[:, ii_x] = e_x_dx[genus-1:-genus+1]
    u_x = u_x_dx


# de-normalise in infinity norm
u_final = u_x_dx[genus-1:-genus+1]
u_final *= u_infty
# de-normalise the reduced field
x_max = propaConfig.N_x * propaConfig.x_step
e_final = u_final / np.sqrt(ConfigSource.k0 * (-ConfigSource.x_s + x_max)) * np.exp(
    1j * ConfigSource.k0 * (-ConfigSource.x_s + x_max))
# e_final = u_x_dx * np.exp(-1j * ConfigSource.k0 * (-ConfigSource.x_s + x_max))

# Field computed by the SSW-2D in SSF
e_ssf = np.load('propagation/outputs/E_field.npy')
if propaConfig.method == 'SSW':
    np.save(
        'SSW_with '+propaConfig.wvl_propagator+'_' + str(propaConfig.x_step * propaConfig.N_x / 1000) + 'km_' + propaConfig.atmosphere,
        e_ssf)
else:
    np.save(propaConfig.method+'_'+str(propaConfig.x_step*propaConfig.N_x/1000)+'km_'+propaConfig.atmosphere, e_ssf)
np.save('WGM_'+str(propaConfig.x_step*propaConfig.N_x/1000)+'km_'+propaConfig.atmosphere, e_final)
np.save('WGM_2d_'+str(propaConfig.x_step*propaConfig.N_x/1000)+'km_'+propaConfig.atmosphere, e_total)