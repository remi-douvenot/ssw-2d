import pywt
import numpy as np
from source.src.complex_source_point import complex_source_point
import csv
import scipy.constants as cst
import matplotlib.pyplot as plt
from propagation.src.propagation.wgm_one_step import compute_connection_coeff


class propaConfig:
    wv_family = 'sym6'
    z_step = 1


j_idx, Lambda_01, Lambda_02 = compute_connection_coeff(propaConfig)
print('j is', j_idx)
print(r'\Lambda_j^{01} is', Lambda_01)
print(r'\Lambda_j^{02} is', Lambda_02)
np.savetxt('Lambda_01.csv', Lambda_01, delimiter=',')
np.savetxt('Lambda_02.csv', Lambda_02, delimiter=',')

plt.figure()
plt.plot(j_idx, Lambda_01)
plt.xlabel('j index', fontsize=14)
plt.ylabel(r'$\Lambda_j^{01}$', fontsize=14)
plt.title('First-order connection coefficient')
plt.grid()
plt.show()

plt.figure()
plt.plot(j_idx, Lambda_02)
plt.xlabel('j index', fontsize=14)
plt.ylabel(r'$\Lambda_j^{02}$', fontsize=14)
plt.title('Second-order connection coefficient')
plt.grid()
plt.show()
