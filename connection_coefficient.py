# Computation of the two-term connection coefficient 02

import pywt
import numpy as np

wav_family = 'db3'
wav = pywt.Wavelet(wav_family)
coeff_h = wav.rec_lo

coeff_a = []
for i in coeff_h:
    coeff_a.append(i * np.sqrt(2))

N = len(coeff_a) # number of filter coefficients
print('Scaling vector is',coeff_a)
print('Length of scaling vector is',N)

j_idx = np.arange(-N+2,N-1) # possible j indixes of the connection coefficient
print('Connection coefficient indexes is', j_idx)

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

print('Connection coefficients 02 are', Lambda_02)

# Computation of the Kronecker delta function

A_delta = np.zeros((2*N-2,2*N-3)) # coefficient matrix of the linear system to solve
eq = 0 # index to number the equation which is created each time

# Generation of the homogeneous equations
for j in j_idx:
    row = np.zeros(2*N-3)
    row[j+N-2] = 1
    for n in range(N):
        for m in range(N):
            if n-m >= -N+2-2*j and n-m <= N-2-2*j:
                idx = np.searchsorted(j_idx , 2*j+n-m)
                row[idx] = row[idx] - 0.5*coeff_a[n]*coeff_a[m]
    A_delta[eq,:] = row
    eq = eq + 1 # increase by 1 to move on to the next equation

# Generation of the non-homogeneous equation
A_delta[eq,:] = np.ones(2*N-3)

B_delta = np.zeros(2*N-3)
B_delta = np.append(B_delta,1)

delta = np.linalg.lstsq(A_delta,B_delta,rcond=None)[0]

print('Kronecker delta function is', delta)