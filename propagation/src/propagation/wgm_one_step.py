import pywt
import numpy as np
import scipy.constants as cst
from scipy.linalg import expm
from propagation.src.atmosphere.genere_n_profile import generate_n_profile


def wgm_one_step(u_x, propagation_matrix):

    # Propagation in one step
    u_dx = np.dot(propagation_matrix, u_x)

    return u_dx


def compute_connection_coeff(propaconfig):
    wav = pywt.Wavelet(propaconfig.wv_family)
    coeff_a = np.array(wav.rec_lo) * np.sqrt(2)

    N = len(coeff_a)  # number of filter coefficients
    z_step = propaconfig.z_step

    # possible j indices of the connection coefficient
    # j_idx = np.around(np.arange(-N + 2, N - 2 + z_step, z_step), 2) # for decimal indices
    j_idx = np.arange(-N + 2, N - 1) # for integer indices
    num_coeff = j_idx.size

    # ---------------------------- #
    # --- First order Lambda01 --- #
    # ---------------------------- #
    A1 = np.zeros((num_coeff + 1, num_coeff))  # coefficient matrix of the linear system to solve
    eq = 0  # index to number the equation which is created each time

    # Generation of the homogeneous equations
    for j in j_idx:
        row = np.zeros(num_coeff)
        idx = np.searchsorted(j_idx, j)
        row[idx] = 1
        for n in range(N):
            for m in range(N):
                if -N + 2 - 2 * j <= n - m <= N - 2 - 2 * j:
                    idx = np.searchsorted(j_idx, round(2 * j + n - m, 2))
                    row[idx] = row[idx] - coeff_a[n] * coeff_a[m]
        A1[eq, :] = row
        eq = eq + 1  # increase by 1 to move on to the next equation

    # Generation of the non-homogeneous equation
    col = 0
    for n in j_idx:
        A1[eq,col] = n
        col = col + 1

    B1 = np.zeros(num_coeff)
    B1 = np.append(B1, 1)

    Lambda_01 = np.linalg.lstsq(A1, B1, rcond=None)[0]

    # ------------ END ----------- #
    # --- First order Lambda01 --- #
    # ---------------------------- #

    # ----------------------------- #
    # --- Second order Lambda02 --- #
    # ----------------------------- #
    A2 = np.zeros((num_coeff + 1, num_coeff))  # coefficient matrix of the linear system to solve
    eq = 0  # index to number the equation which is created each time

    # Generation of the homogeneous equations
    for j in j_idx:
        row = np.zeros(num_coeff)
        idx = np.searchsorted(j_idx, j)
        row[idx] = 1
        for n in range(N):
            for m in range(N):
                if -N + 2 - 2 * j <= n - m <= N - 2 - 2 * j:
                    idx = np.searchsorted(j_idx, round(2 * j + n - m, 2))
                    row[idx] = row[idx] - 2 * coeff_a[n] * coeff_a[m]
        A2[eq, :] = row
        eq = eq + 1  # increase by 1 to move on to the next equation

    # Generation of the non-homogeneous equation
    col = 0
    for j in j_idx:
        coeff_j = 0
        for k in range(N):
            coeff_j = coeff_j + j * k * coeff_a[k]
        coeff_j = coeff_j + j ** 2
        A2[eq, col] = coeff_j
        col = col + 1

    B2 = np.zeros(num_coeff)
    B2 = np.append(B2, 2) # for integer indices
    # B2 = np.append(B2, 2 / z_step) # for decimal indices

    Lambda_02 = np.linalg.lstsq(A2, B2, rcond=None)[0]

    # ------------ END ------------ #
    # --- Second order Lambda02 --- #
    # ----------------------------- #

    return j_idx, Lambda_01, Lambda_02


def galerkin_matrices(propaconfig):
    k0 = 2 * cst.pi * propaconfig.freq / cst.c
    z_max = propaconfig.N_z * propaconfig.z_step
    wav = pywt.Wavelet(propaconfig.wv_family)
    genus = wav.number
    sup_len = propaconfig.N_z + 2 * (genus - 1)  # length of support of the extended domain

    j_idx, Lambda01, Lambda02 = compute_connection_coeff(propaconfig)

    # Generate n profile
    n_refractive_index = generate_n_profile(propaconfig)

    # delta, L and S matrices (Iqbal)
    # delta_matrix = np.eye(propaconfig.N_z, dtype='complex')
    L_matrix = np.zeros((sup_len, sup_len), dtype='complex')

    for diag in j_idx:
        L_matrix = L_matrix + np.diag(np.ones(sup_len - np.abs(int(diag/propaconfig.z_step))) * 1j / (2 * k0) *
                                      Lambda02[int(diag + (j_idx.size-1)/2)], k=int(diag/propaconfig.z_step))

    n_refractive_index_up = n_refractive_index[-genus - 1:-1]
    n_refractive_index_down = n_refractive_index[0:genus - 2]
    n_refractive_index = np.concatenate((n_refractive_index_down, n_refractive_index, n_refractive_index_up))

    S_matrix = np.diag(1j * k0 / 2 * (n_refractive_index ** 2 - 1))

    # matrix for propagation in delta_x (Iqbal)
    '''num = 2 * delta_matrix - (L_matrix + S_matrix) * propaconfig.x_step
    den = np.linalg.inv(2 * delta_matrix + (L_matrix + S_matrix) * propaconfig.x_step)
    propagation_matrix = np.dot(den, num)'''
    propagation_matrix = expm(- (L_matrix + S_matrix) * propaconfig.x_step)

    return L_matrix, S_matrix, propagation_matrix
