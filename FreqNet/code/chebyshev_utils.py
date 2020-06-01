# -*- coding: utf-8 -*-
# File: chebyshev_utils.py

import numpy as np

def chebyshev_func(n: int, x):
    return np.cos(n * np.arccos(x))


def construct_chebyshev_basis(N: int):
    """
    0/N,...,N-1/N value points (value domain: 0-1)
    """
    w_all = []
    for n in range(1, N+1):
        w = np.array([chebyshev_func(n, i / N) for i in range(N)])
        w_all.append(w)

    return w_all


def get_w_chebyshev(X, l, labels, patch_size, chebyshev_basis_sel='variance'):
    k1, k2 = patch_size
    N = k1 * k2
    assert N == X.shape[0]
    w_all = construct_chebyshev_basis(N)

    if chebyshev_basis_sel in ['variance', 'magnitude']:
        w_stats = []
        for one_w in w_all:
            one_w_unit = one_w / np.linalg.norm(one_w)
            one_w_prod = np.matmul(np.asarray(one_w_unit).reshape((1, N)), X)
            w_stats.append((one_w_unit, np.mean(abs(one_w_prod)), np.std(one_w_prod)))
        if chebyshev_basis_sel == 'variance':
            # sort w by variance in descending order
            w_stats.sort(key=lambda tup: tup[2], reverse=True)
        elif chebyshev_basis_sel == 'magnitude':
            w_stats.sort(key=lambda tup: tup[1], reverse=True)

        w = [np.asarray(tup[0]).reshape(k1, k2) for tup in w_stats[:l]]
    else:  # group method
        raise NotImplementedError
    return w
