import pywt
import numpy as np
from source.src.complex_source_point import complex_source_point
import csv
import scipy.constants as cst
import matplotlib.pyplot as plt
from propagation.src.propagation.connection_coefficient_one_step import connection_coefficient_one_step

# --------------------- #
# --- INITIAL FIELD --- #
# --------------------- #
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
        ConfigSource.n_z = np.int64(row[1])                                                                       #change
    elif row[0] == 'z_step':
        ConfigSource.z_step = np.float64(row[1])
    elif row[0] == 'x_s':  # position along x --> must be <0
        ConfigSource.x_s = np.float64(row[1])
    elif row[0] == 'type':
        source_type = row[1]
    elif row[0] == 'frequency':
        freq = np.float64(row[1])*1e6
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
        raise ValueError(['Input file of the geometry is not valid. Input "' + row[0]+'" not valid'])

# wavenumber
ConfigSource.k0 = 2 * cst.pi * freq / cst.c
# compute E field
e_field = complex_source_point(ConfigSource) # initial field

# plot E_init
plt.figure()
ax = plt.subplot(111)
e_field_db = 20 * np.log10(np.abs(e_field))
v_max = np.max(e_field_db)
v_min = v_max - 100
print('Max field = ', np.round(v_max, 2), 'dBV/m')
z_vect = np.linspace(0, ConfigSource.z_step*ConfigSource.n_z, num=ConfigSource.n_z, endpoint=False)
plt.plot(e_field_db, z_vect)
plt.xlim(v_min, v_max)
plt.ylim(0, ConfigSource.z_step*ConfigSource.n_z)
plt.xlabel('E field (dBV/m)', fontsize=14)
plt.ylabel('Altitude (m)', fontsize=14)
plt.title('Initial field E')
plt.grid()
plt.show()

# --- Calculate u_0 from E_init (normalised in infinity norm to have max(|u_0|) = 1) --- #
u_0 = e_field * np.sqrt(ConfigSource.k0 * (-ConfigSource.x_s)) * np.exp(1j * ConfigSource.k0 * (-ConfigSource.x_s))
# u_0 = e_field / np.sqrt(k0*(-config.x_s))
u_infty = np.max(np.abs(u_0))  # norm infinity of the initial field
u_0 /= u_infty  # put max at 1 to avoid numerical errors

# --------- END ------- #
# --- INITIAL FIELD --- #
# --------------------- #

delta_x = 100
u_dx = connection_coefficient_one_step(u_0,'sym6',delta_x,ConfigSource)

# de-normalise in infinity norm
u_dx *= u_infty
# de-normalise the reduced field
e_dx = u_dx / np.sqrt(ConfigSource.k0 * (-ConfigSource.x_s + delta_x)) * np.exp(-1j * ConfigSource.k0 * (-ConfigSource.x_s + delta_x))

# Field computed by the SSW-2D
e_ssw_2d = np.load('propagation/outputs/E_field.npy')

print(e_dx)
print(e_ssw_2d)

# Plot the field after propagation
plt.figure()
ax = plt.subplot(111)

e_dx_db = 20 * np.log10(np.abs(e_dx))
e_ssw_2d_db = 20 * np.log10(np.abs(e_ssw_2d))

v_max = np.max(e_dx_db)
v_min = v_max - 100
print('Max field = ', np.round(v_max, 2), 'dBV/m')

plt.plot(e_field_db, z_vect, color='g', label='Initial field')
plt.plot(e_dx_db, z_vect, color='r', label='Connection coefficients')
plt.plot(e_ssw_2d_db, z_vect, color='b', label='SSF')
plt.xlim(v_min, v_max)
plt.ylim(0, ConfigSource.z_step * ConfigSource.n_z)
plt.xlabel('E field (dBV/m)', fontsize=14)
plt.ylabel('Altitude (m)', fontsize=14)
plt.title('Final field E')
plt.legend()
plt.grid()
plt.show()

'''error = e_dx_db - e_ssw_2d_db
plt.figure()
plt.plot(error,z_vect)
plt.xlabel('Error (dBV/m)',fontsize=14)
plt.ylabel('Altitude (m)', fontsize=14)
plt.grid()
plt.show()'''