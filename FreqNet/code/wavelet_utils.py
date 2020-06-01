import numpy as np
import math
import pywt
from tqdm import tqdm

def c_matmul(A, B):
    res_shape = (A.shape[0] * B.shape[0], A.shape[1] * B.shape[1])
    res = np.zeros(res_shape)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            for k in range(B.shape[0]):
                for l in range(B.shape[1]):
                    val = A[i][j] * B[k][l]
                    res[i*B.shape[0]+k][j*B.shape[1]+l] = val
    return res

# Haar basis
def H(N, j):
    '''
    N: int (should be power of 2), length of signal
    j: int, scale
    '''
    print('(N, j)', N, j)
    max_scale = math.log(N, 2)
    print('max_scale:', int(max_scale))
    if max_scale - abs(max_scale) >= 1e-6:
        raise ValueError
    if max_scale-1 < j:
        raise ValueError
    # base case
    if max_scale-1 == j:
        scale_coef = np.array([1/math.sqrt(2), 1/math.sqrt(2)]).reshape(1, -1)
        wavelet_coef = np.array([1/math.sqrt(2), -1/math.sqrt(2)]).reshape(1, -1)
        I = np.eye(int(N/2))
        ret = np.concatenate([c_matmul(I, scale_coef), c_matmul(I, wavelet_coef)], axis=0)
    else:
        ret = H(1 << (j+1), j)
        for jn in range(j+1, int(max_scale)):
           h = H(1 << (jn+1), jn)
           h_lower = h[:h.shape[0]//2, :]
           h_upper = h[h.shape[0]//2:, :]
           ret = np.concatenate([np.matmul(ret, h_lower), h_upper], axis=0)
    return ret

def get_conv_res_wavelets(X, sel_idx, wavelet='db1', level=1):
    sel_idx_a, sel_idx_d = sel_idx
    wave_coefs_a = []
    wave_coefs_d = []
    number_of_patches, _ = X.shape
    
    for i in range(number_of_patches):
        curr_patch = X[i]
        curr_coefs = pywt.wavedecn(curr_patch, wavelet=wavelet, level=level, mode='per')
        curr_coefs, coefs_slices = pywt.coeffs_to_array(curr_coefs)
        curr_a = curr_coefs[coefs_slices[0]]
        wave_coefs_a.append(curr_a)
        wave_coefs_d.append(curr_coefs[len(curr_a):])
    if sel_idx_a is not None:
        wave_coefs_a = np.asarray(wave_coefs_a)
        wave_coefs_d = np.asarray(wave_coefs_d)

        wave_conv_res_a = wave_coefs_a[:, sel_idx_a]
        wave_conv_res_d = wave_coefs_d[:, sel_idx_d]

        return np.concatenate([wave_conv_res_a, wave_conv_res_d], axis=1)
    else:
        wave_coefs_d = np.asarray(wave_coefs_d)
        wave_conv_res_d = wave_coefs_d[:, sel_idx_d]

        return wave_conv_res_d

def get_coeffs_idx_wavelets(X, l_a, l_d, wavelet='db1', level=1, basis_sel='magnitude'):
    # approximation
    wave_coefs_a = []
    # detail
    wave_coefs_d = []

    number_of_patches, _ = X.shape
    
    for i in tqdm(range(number_of_patches)):
        curr_patch = X[i]
        curr_coefs = pywt.wavedecn(curr_patch, wavelet=wavelet, level=level, mode='per')
        curr_coefs, coefs_slices = pywt.coeffs_to_array(curr_coefs)
        curr_a = curr_coefs[coefs_slices[0]]
        wave_coefs_a.append(curr_a)
        wave_coefs_d.append(curr_coefs[len(curr_a):])

    wave_coefs_a = np.asarray(wave_coefs_a)
    wave_coefs_d = np.asarray(wave_coefs_d)

    # in the case of number of approximations is less than l_a,
    # we will select all the approximations
    if len(wave_coefs_a) < l_a:
        l_d += len(wave_coefs_a) - l_a
        l_a = len(wave_coefs_a)

    if l_a == len(wave_coefs_a):
        selected_coef_a = np.arange(len(wave_coefs_a))
    else:
        if basis_sel == 'variance':
            coefs_a_var = np.var(wave_coefs_a, axis=0)
            coefs_a_var = sorted([(i, coefs_a_var[i]) for i in range(coefs_a_var.size)], key=lambda x: x[1], reverse=True)
            selected_coef_a = np.asarray([tup[0] for i, tup in enumerate(coefs_a_var) if i < l_a])
        elif basis_sel == 'magnitude':
            coefs_a_mean = np.mean(np.abs(wave_coefs_a), axis=0)
            coefs_a_mean = sorted([(i, coefs_a_mean[i]) for i in range(coefs_a_mean.size)], key=lambda x: x[1], reverse=True)
            selected_coef_a = np.asarray([tup[0] for i, tup in enumerate(coefs_a_mean) if i < l_a])
    
    if basis_sel == 'variance':
        # assume l_d < number of details
        coefs_d_var = np.var(wave_coefs_d, axis=0)
        coefs_d_var = sorted([(i, coefs_d_var[i]) for i in range(coefs_d_var.size)], key=lambda x: x[1], reverse=True)
        selected_coef_d = np.asarray([tup[0] for i, tup in enumerate(coefs_d_var) if i < l_d])
    elif basis_sel == 'magnitude':
        # assume l_d < number of details
        coefs_d_mean = np.mean(np.abs(wave_coefs_d), axis=0)
        coefs_d_mean = sorted([(i, coefs_d_mean[i]) for i in range(coefs_d_mean.size)], key=lambda x: x[1], reverse=True)
        selected_coef_d = np.asarray([tup[0] for i, tup in enumerate(coefs_d_mean) if i < l_d])

    if len(selected_coef_a) > 0:
        wave_conv_res_a = wave_coefs_a[:, selected_coef_a]
        wave_conv_res_d = wave_coefs_d[:, selected_coef_d]
        return  (selected_coef_a, selected_coef_d), np.concatenate([wave_conv_res_a, wave_conv_res_d], axis=1)
    else:
        wave_conv_res_d = wave_coefs_d[:, selected_coef_d]
        return (None, selected_coef_d), wave_conv_res_d

def wavedec_patch(X, level, wavelet='db1'):
    coefs = []
    for i in range(X.shape[0]):
        curr_coef = pywt.wavedec(X[i], wavelet=wavelet, level=level, mode='per')
        coefs.append(curr_coef)
    return coefs

if __name__ == '__main__':
    print(H(8, 0))
