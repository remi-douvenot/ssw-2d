import numpy as np
import matplotlib.pyplot as plt
import sys

x_max = 1.0  # km
z_step = 0.2  # m
n_z = 2000
z_max = n_z * z_step  # m
atmosphere = 'Homogeneous'

e_ssf = np.load('SSF_'+str(x_max)+'km_'+atmosphere+'.npy')
e_ssw_dssf = np.load('SSW_with DSSF_'+str(x_max)+'km_'+atmosphere+'.npy')
e_wgm = np.load('WGM_'+str(x_max)+'km_'+atmosphere+'.npy')
e_ssw_wgm = np.load('SSW_with WGM_'+str(x_max)+'km_'+atmosphere+'.npy')

e_ssf_db = 20 * np.log10(np.abs(e_ssf))
e_ssw_dssf_db = 20 * np.log10(np.abs(e_ssw_dssf))
e_wgm_db = 20 * np.log10(np.abs(e_wgm))
e_ssw_wgm_db = 20 * np.log10(np.abs(e_ssw_wgm))

# Plot the field after propagation
plt.figure()
v_max = np.max([e_ssf_db, e_ssw_dssf_db, e_wgm_db])
v_min = v_max - 100
print('Max output field = ', np.round(v_max, 2), 'dBV/m')
z_vect = np.linspace(0, z_step * n_z, num=n_z, endpoint=False)
plt.plot(e_wgm_db, z_vect, color='r', label='WGM')
plt.plot(e_ssf_db, z_vect, color='b', label='SSF')
plt.plot(e_ssw_dssf_db, z_vect, color='g', label='SSW')
plt.xlim(v_min, v_max+10)
plt.ylim(0, z_max)
plt.xlabel('E field (dBV/m)', fontsize=14)
plt.ylabel('Altitude (m)', fontsize=14)
plt.title('Final field E at '+str(x_max)+' km for '+atmosphere+' atmosphere')
plt.legend()
plt.grid()
plt.show()

# E field WGM in (x,z) plane
e_total = np.load('WGM_2d_'+str(x_max)+'km_'+atmosphere+'.npy')
plt.figure()
plot_dynamic = 80  # dB from max to min
v_max = np.max(20 * np.log10(np.abs(e_total) + sys.float_info.epsilon))
v_min = v_max - plot_dynamic
extent = [0, x_max, 0, z_max]
im_field = plt.imshow(20 * np.log10(abs(e_total) + sys.float_info.epsilon),
                      extent=extent, aspect='auto', vmax=v_max,
                      vmin=v_min, origin='lower', interpolation='none', cmap='jet')
cb = plt.colorbar(im_field)
cb.ax.tick_params(labelsize=12)
plt.xlabel('Distance (km)', fontsize=14)
plt.ylabel('Altitude (m)', fontsize=14)
plt.show()

# SSW dictionary with DSSF vs with WGM
plt.figure()
v_max = np.max([e_ssw_wgm_db, e_ssw_dssf_db])
v_min = v_max - 100
plt.plot(e_ssw_wgm_db, z_vect, color='r', label='With WGM')
plt.plot(e_ssw_dssf_db, z_vect, color='b', label='With DSSF')
plt.xlim(v_min, v_max+10)
plt.ylim(0, z_max)
plt.xlabel('E field (dBV/m)', fontsize=14)
plt.ylabel('Altitude (m)', fontsize=14)
plt.title('SSW propagation at '+str(x_max)+' km for '+atmosphere+' atmosphere')
plt.legend()
plt.grid()
plt.show()