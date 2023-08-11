import pywt
import numpy as np
from source.src.complex_source_point import complex_source_point
import csv
import scipy.constants as cst
import matplotlib.pyplot as plt
from propagation.src.propagation.connection_coefficient_one_step import connection_coefficient_one_step, galerkin_matrices
from propagation.src.propagation.apodisation import apply_apodisation, apodisation_window
import sys

# ----------------------------- #
# --- Extract configuration --- #
# ----------------------------- #
'''file_source = 'source/inputs/configuration.csv'


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


ConfigSource.k0 = 2 * cst.pi * freq / cst.c'''


'''file_config_propa = 'propagation/inputs/configuration.csv'


class propaConfig:
    def __init__(self):
        self.wv_family = 0
        self.x_step = 0
        self.N_z = 0
        self.z_step = 0
        self.atmosphere = 0
        self.freq = 0


f_propa = open(file_config_propa, newline='')
propa_tmp = csv.reader(f_propa)
for row in propa_tmp:
    if row[0] == 'N_z':
        propaConfig.N_z = np.int64(row[1])
    elif row[0] == 'N_x':
        propaConfig.N_x = np.int64(row[1])
    elif row[0] == 'z_step':
        propaConfig.z_step = np.float64(row[1])
    elif row[0] == 'x_step':
        propaConfig.x_step = np.float64(row[1])
    elif row[0] == 'wavelet family':
        propaConfig.wv_family = row[1]
    elif row[0] == 'atmosphere':
        propaConfig.atmosphere = row[1]
    elif row[0] == 'frequency':
        propaConfig.freq = np.float64(row[1]) * 1e6
    else:
        pass'''


class propaConfig:
    freq = 1000e6
    wv_family = 'sym6'
    x_step = 0.1
    N_x = 1000
    z_step = 0.2
    N_z = 2000
    atmosphere = 'Homogeneous'
    c0 = 0.118
    delta = 15.0
    zb = 200.0
    c2 = -0.8
    zt = 200.0
    apo_z = 0.2
    apo_window = 'Hanning'
    ground = 'No Ground'


class ConfigSource:
    n_z = 2000
    z_step = 0.2
    # position of the CSP along x (real <0)
    x_s = -50
    # radiated power in W (real)
    P_Tx = 1
    # max gain in dBi (real)
    G_Tx = 0
    # wavenumber in m
    k0 = 2 * cst.pi * propaConfig.freq / cst.c
    # altitude of the source in m
    z_s = 200
    # width of the aperture W0
    W0 = cst.c / propaConfig.freq * 4

# ------------ END ------------ #
# --- Extract configuration --- #
# ----------------------------- #


# --------------------- #
# --- INITIAL FIELD --- #
# --------------------- #

# compute E field
e_field = complex_source_point(ConfigSource)  # initial field

# plot E_init
plt.figure()
e_field_db = 20 * np.log10(np.abs(e_field))
v_max = np.max(e_field_db)
v_min = v_max - 100
print('Max input field = ', np.round(v_max, 2), 'dBV/m')
z_vect = np.linspace(0, ConfigSource.z_step * ConfigSource.n_z, num=ConfigSource.n_z, endpoint=False)
plt.plot(e_field_db, z_vect)
plt.xlim(v_min, v_max+1)
plt.ylim(0, ConfigSource.z_step * ConfigSource.n_z)
plt.xlabel('E field (dBV/m)', fontsize=14)
plt.ylabel('Altitude (m)', fontsize=14)
plt.title('Initial field E')
plt.grid()
plt.show()

# --- Calculate u_0 from E_init (normalised in infinity norm to have max(|u_0|) = 1) --- #
u_0 = e_field * np.sqrt(ConfigSource.k0 * (-ConfigSource.x_s)) * np.exp(-1j * ConfigSource.k0 * (-ConfigSource.x_s))
# u_0 = e_field * np.exp(1j * ConfigSource.k0 * (-ConfigSource.x_s))
u_infty = np.max(np.abs(u_0))  # norm infinity of the initial field
u_0 /= u_infty  # put max at 1 to avoid numerical errors

# -------- END -------- #
# --- INITIAL FIELD --- #
# --------------------- #

e_total = np.zeros((propaConfig.N_z, propaConfig.N_x), dtype='complex')
n_apo_z = np.int64(propaConfig.apo_z * propaConfig.N_z)
apo_window_z = apodisation_window(propaConfig.apo_window, n_apo_z)
u_x = u_0

# Propagation
L_matrix, S_matrix, propagation_matrix = galerkin_matrices(propaConfig)

for ii_x in np.arange(0, propaConfig.N_x):
    u_x = apply_apodisation(u_x, apo_window_z, propaConfig)
    u_x_dx = connection_coefficient_one_step(u_x, propagation_matrix)
    e_x_dx = u_x_dx * u_infty / np.sqrt(ConfigSource.k0 * (-ConfigSource.x_s + ii_x * propaConfig.x_step)) * np.exp(
        1j * ConfigSource.k0 * (-ConfigSource.x_s + ii_x * propaConfig.x_step))
    e_total[:, ii_x] = e_x_dx
    u_x = u_x_dx


# de-normalise in infinity norm
u_x_dx *= u_infty
# de-normalise the reduced field
x_max = propaConfig.N_x * propaConfig.x_step
e_final = u_x_dx / np.sqrt(ConfigSource.k0 * (-ConfigSource.x_s + x_max)) * np.exp(
    1j * ConfigSource.k0 * (-ConfigSource.x_s + x_max))
# e_final = u_x_dx * np.exp(-1j * ConfigSource.k0 * (-ConfigSource.x_s + x_max))

# Field computed by the SSW-2D in SSF
e_ssf = np.load('propagation/outputs/E_field.npy')

# Plot the field after propagation
plt.figure()
e_final_db = 20 * np.log10(np.abs(e_final))
e_ssf_db = 20 * np.log10(np.abs(e_ssf))
v_max = np.max(e_final_db)
v_min = v_max - 100
print('Max output field = ', np.round(v_max, 2), 'dBV/m')
z_vect = np.linspace(0, ConfigSource.z_step * ConfigSource.n_z, num=ConfigSource.n_z, endpoint=False)
# plt.plot(e_field_db, z_vect, color='g', label='Initial field')
plt.plot(e_final_db, z_vect, color='r', label='Wavelet-Galerkin')
z_vect2 = np.linspace(0, 400, num=2000, endpoint=False)
plt.plot(e_ssf_db, z_vect2, color='b', label='SSF')
plt.xlim(v_min, -15)
plt.ylim(0, ConfigSource.z_step * ConfigSource.n_z)
plt.xlabel('E field (dBV/m)', fontsize=14)
plt.ylabel('Altitude (m)', fontsize=14)
plt.title('Final field E')
plt.legend()
plt.grid()
plt.show()



plt.figure()
plot_dynamic = 60 # dB from max to min
v_max = np.max(20 * np.log10(np.abs(e_total) + sys.float_info.epsilon))
v_min = v_max - plot_dynamic
z_max = propaConfig.N_z * propaConfig.z_step
extent = [0, x_max, 0, z_max]
im_field = plt.imshow(20 * np.log10(abs(e_total) + sys.float_info.epsilon),
                      extent=extent, aspect='auto', vmax=v_max,
                      vmin=v_min, origin='lower', interpolation='none', cmap='jet')
cb = plt.colorbar(im_field)
cb.ax.tick_params(labelsize=12)
plt.xlabel('Distance (m)', fontsize=14)
plt.ylabel('Altitude (m)', fontsize=14)
plt.title('Propagated field u')
plt.show()