import pywt
import numpy as np
import scipy.constants as cst
from propagation.src.propagation.wgm_one_step import wgm_one_step, galerkin_matrices
from propagation.src.propagation.apodisation import apply_apodisation, apodisation_window
from src.wavelets.wavelet_operations import sparsify


def wgm_2d(u_0, config, n_refraction, ii_vect_relief):

    n_x = config.N_x

    # --- Extended domain --- #
    print('Extending domain')
    wav = pywt.Wavelet(config.wv_family)
    genus = wav.number
    ext_dom = np.zeros(genus - 1)
    u_x = np.concatenate((ext_dom, u_0, ext_dom))
    sup_len = config.N_z + 2 * (genus - 1)  # length of support of the extended domain
    n_refraction_up = n_refraction[-genus - 1:-1]
    n_refraction_down = n_refraction[0:genus - 2]
    n_refraction = np.concatenate((n_refraction_down, n_refraction, n_refraction_up))
    # ----------------------- #

    # --- Apodisation window --- #
    n_apo_z = np.int64(config.apo_z * config.N_z + (genus - 1))
    apo_window_z = apodisation_window(config.apo_window, n_apo_z)
    # -------------------------- #

    # --- Propagation matrix --- #
    print('Computing propagation matrix')
    propagation_matrix = galerkin_matrices(config, sup_len, n_refraction)
    # -------------------------- #

    wv_total = [[]] * n_x

    for ii_x in np.arange(1, n_x+1):
        if ii_x % 100 == 0:
            print('Iteration', ii_x, '/', n_x, '. Distance =', ii_x*config.x_step)

        # Apodisation
        u_x = apply_apodisation(u_x, apo_window_z, config)

        # Propagation
        u_x_dx = wgm_one_step(u_x, propagation_matrix)

        # update u_x
        u_x = u_x_dx

        # store field as a wavelet decomposition
        wv_total[ii_x - 1] = sparsify(pywt.wavedec(u_x[genus-1:-genus+1], config.wv_family, 'per', config.wv_L))

    u_final = u_x_dx[genus - 1:-genus + 1]

    return u_final, wv_total