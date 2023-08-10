import pywt
import numpy as np
from source.src.complex_source_point import complex_source_point
import csv
import scipy.constants as cst
import matplotlib.pyplot as plt
from propagation.src.propagation.connection_coefficient_one_step import connection_coefficient_one_step, galerkin_matrices, compute_connection_coeff


class propaConfig:
    freq = 1000e6
    wv_family = 'sym6'
    x_step = 2
    N_x = 100
    z_step = 1
    N_z = 2000
    atmosphere = 'Homogeneous'


j_idx, Lambda_01, Lambda_02 = compute_connection_coeff(propaConfig)

plt.figure()
plt.plot(j_idx, Lambda_01)
plt.xlabel('j index', fontsize=14)
plt.ylabel('Lambda_01', fontsize=14)
plt.title('First-order connection coefficient')
plt.grid()
plt.show()

plt.figure()
plt.plot(j_idx, Lambda_02)
plt.xlabel('j index', fontsize=14)
plt.ylabel('Lambda_02', fontsize=14)
plt.title('Second-order connection coefficient')
plt.grid()
plt.show()
