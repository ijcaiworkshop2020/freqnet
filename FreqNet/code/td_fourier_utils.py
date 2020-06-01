import numpy as np
import math
from fourier_utils import get_w_fourier_group_method

def construct_2d_fourier_basis(N: int, k1: int, k2: int, complex: bool=False):
    one_period_k1 = 2 * np.pi / k1
    one_period_k2 = 2 * np.pi / k2
    w_all = []
    if not complex:
        for k in range((math.floor(k1 / 2) + 1)):
            for l in range((math.floor(k2 / 2) + 1)):
                cos_array = []
                sin_array = []
                for i in range(k1):
                    for j in range(k2):
                        cos_array.append(np.cos((k*i*one_period_k1+l*j*one_period_k2))/math.sqrt(N))
                        sin_array.append(np.sin(-(k*i*one_period_k1+l*j*one_period_k2+1))/math.sqrt(N))
                w_all.append(cos_array)
                w_all.append(sin_array)
    else:
        for k in range((math.floor(k1 / 2) + 1)):
            for l in range((math.floor(k2 / 2) + 1)):
                triangle_array = []
                for i in range(k1):
                    for j in range(k2):
                        triangle_array.append(np.complex(np.cos((k*i*one_period_k1+l*j*one_period_k2))/math.sqrt(N), np.sin(-(k*i*one_period_k1+l*j*one_period_k2))/math.sqrt(N)))
                w_all.append(triangle_array)

    return w_all

def get_w_2d_fourier(X, l, labels, patch_size, fourier_basis_sel='variance', rgb=False):
    # N = k1 * k2
    # cosin basis:
    #  c_k = {cos(2*k*pi/N * 0), ..., cos(2*k*pi/N * (N-1))}/sqrt(N)
    # sin basis:
    #  s_k = {sin(2*k*pi/N * 0), ..., sin(2*k*pi/N * (N-1))}/sqrt(N)
    # where
    # for c_k, k in {0, 1, ..., trunc(N/2)}
    # for s_k, k in {-trunc((N-1)/2), ..., -1)}
    k1, k2 = patch_size
    # assert N == X.shape[0]
    if rgb:
        N = k1 * k2 * 3
    else:
        N = k1 * k2
    w_all = construct_2d_fourier_basis(N, k1, k2)  # w_all=[cos,sin]

    # w_all contains all the fourier basis for N-dim space,
    # then we try to find "the most important" l base vectors:
    #     1. calculate the inner products of each basis and each patch
    #     2. get abs mean and variance of the inner products
    #     3. pick top l basis (variance? magnitude?)
    if fourier_basis_sel in ['variance', 'magnitude']:
        w_stats = []
        for one_w in w_all:
            one_w_unit = one_w / np.linalg.norm(one_w)
            one_w_array = np.asarray(one_w_unit).reshape((1, -1))
            one_w_prod = np.matmul(one_w_array, X)
            # one_w_prod = np.matmul(np.asarray(one_w_unit).reshape((1, -1)), X)
            w_stats.append((one_w_unit, np.mean(abs(one_w_prod)), np.std(one_w_prod)))
        if fourier_basis_sel == 'variance':
            # sort w by variance in descending order
            w_stats.sort(key=lambda tup: tup[2], reverse=True)
        elif fourier_basis_sel == 'magnitude':
            w_stats.sort(key=lambda tup: tup[1], reverse=True)

        w = [np.asarray(tup[0]).reshape(k1, k2, -1) for tup in w_stats[:l]]
    else:  # group method
        w = get_w_fourier_group_method(X, l, labels, patch_size, fourier_basis_sel, w_all)
    w = [np.squeeze(item) for item in w]
    return w


if __name__ == '__main__':
    w = construct_2d_fourier_basis(49, 7, 7, False)
    for one_w in w:
        print(one_w)