import pywt
import numpy as np

def connection_coefficient_one_step(u_x,wav_family,delta_x,config):
    j_idx, Lambda02 = compute_connection_coeff(wav_family)

    # delta and L matrices (Iqbal)
    delta_matrix = np.eye(config.n_z, dtype='complex')
    L_matrix = np.zeros((config.n_z, config.n_z), dtype='complex')
    M_matrix = np.zeros((config.n_z, config.n_z), dtype='complex')

    for k in range(config.n_z):
        for l in range(config.n_z):
            j = l - k
            if j >= min(j_idx) and j <= max(j_idx):
                idx = np.searchsorted(j_idx, j)
                L_matrix[k, l] = 1j * Lambda02[idx] / (2 * config.k0)
            else:
                L_matrix[k, l] = 0

    M_matrix = (2 * delta_matrix - L_matrix * delta_x) * np.linalg.inv(2 * delta_matrix + L_matrix * delta_x)  # matrix for propagation in delta_x (Iqbal)

    # Propagation in one step
    coeff_u = pywt.wavedec(u_x, wav_family, level=0)  # approximation coefficients
    coeffs_arr, coeffs_slice = pywt.coeffs_to_array(coeff_u)
    coeffs_arr_dx = np.dot(M_matrix, coeffs_arr)
    coeff_u_dx = pywt.array_to_coeffs(coeffs_arr_dx, coeffs_slice, output_format='wavedec')
    u_dx = pywt.waverec(coeff_u_dx, wav_family)

    return u_dx


def compute_connection_coeff(wav_family):
    wav = pywt.Wavelet(wav_family)
    coeff_h = wav.rec_lo

    coeff_a = []
    for i in coeff_h:
        coeff_a.append(i * np.sqrt(2))

    N = len(coeff_a) # number of filter coefficients

    j_idx = np.arange(-N+2,N-1) # possible j indixes of the connection coefficient

    A = np.zeros((2*N-2,2*N-3)) # coefficient matrix of the linear system to solve
    eq = 0 # index to number the equation which is created each time

    # Generation of the homogeneous equations
    for j in j_idx:
        row = np.zeros(2*N-3)
        row[j+N-2] = 1
        for n in range(N):
            for m in range(N):
                if n-m >= -N+2-2*j and n-m <= N-2-2*j:
                    idx = np.searchsorted(j_idx , 2*j+n-m)
                    row[idx] = row[idx] - 2*coeff_a[n]*coeff_a[m]
        A[eq,:] = row
        eq = eq + 1 # increase by 1 to move on to the next equation

    # Generation of the non-homogeneous equation
    col = 0
    for j in j_idx:
        coeff_j = 0
        for k in range(N):
            coeff_j = coeff_j + j*k*coeff_a[k]
        coeff_j = coeff_j + j**2
        A[eq,col] = coeff_j
        col = col + 1

    B = np.zeros(2*N-3)
    B = np.append(B,2)

    Lambda_02 = np.linalg.lstsq(A,B,rcond=None)[0]

    return j_idx, Lambda_02