# -*- coding: utf-8 -*-
# File: basic_utils.py

import numpy as np
import os
from os import path
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import StandardScaler
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.image import extract_patches_2d
from tqdm import tqdm
from collections import Counter, namedtuple
from scipy import sparse
import random
import logger  # vscode static check yields errors, but it is fine in runtime. <https://gist.github.com/Yeeef/f47faa3bd8e93a4fbcd902c6dcfe7fb1>
import tkinter
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

def randomized_pca(X, n_components, rank=1000, with_std=False, q=1):
    """
    see <https://arxiv.org/pdf/0909.4061.pdf> for more details
    Args:
        * X: n_samples * n_features (n * m)
        * n_components: number of eigenvectors needed (l)
        * rank: k
        * std: whether to standarize X feature-wise
        * q: exponent param
    """
    assert len(X.shape) == 2, X.shape
    assert X.shape[0] > X.shape[1], 'X should be n_samples * n_features, and currently we only support n_samples > n_features'
    X = StandardScaler(with_std=with_std).fit_transform(X).T  # convert to m * n, thus the u vector of XX' is the eigenvector of XX'
    m, n = X.shape
    Omega = np.random.randn(n, rank)

    Y = (X @ X.T) @ (X @ X.T) @ X @ Omega if q == 2 else (X @ X.T) @ X @ Omega  # Y: m * 2k
    Q, *_ = np.linalg.svd(Y, full_matrices=False)  # Q: m * 2k
    B = Q.T @ X  # 2k * n
    ub, *_ = np.linalg.svd(B, full_matrices=False)  # 2k * 2k
    u = Q @ ub  # m * 2k
    return u.T[:n_components, ...]  # l * m


def get_mnist_basic_data(base_dir, n_imgs=None, train=True, shuffle=False, log=False):
    """
    mnist basic data, train 12000, test 50000
    can also be used to get the other variational mnist data
    https://sites.google.com/a/lisa.iro.umontreal.ca/public_static_twiki/variations-on-the-mnist-digits
    """
    if train:
        dataset = np.loadtxt(path.join(base_dir, 'mnist_train.amat'))
    else:
        dataset = np.loadtxt(path.join(base_dir, 'mnist_test.amat'))
    indices = list(range(len(dataset)))
    if shuffle:
        random.shuffle(indices)
    if n_imgs is not None:
        assert n_imgs <= len(indices)
        dataset = dataset[indices[:n_imgs]]
    data_points = dataset[:, :-1].reshape((-1, 28, 28))
    labels = dataset[:, -1]
    send_msg(log, "max of data: {}, min of data: {}, mean of data: {}".format(np.max(data_points), np.min(data_points), np.mean(data_points)))
    return data_points, labels


def grayscale(data, dtype='float32'):
    '''
    Luma coding
    '''
    r, g, b = np.asarray(0.3, dtype=dtype), np.asarray(0.59, dtype=dtype), np.asarray(0.11, dtype=dtype)
    rst = r * data[:, :, :, 0] + g * data[:, :, :, 1] + b * data[:, :, :, 2]
    return rst

def show_cifar10_image(img):
    plt.figure(figsize=(4, 2))
    plt.subplot(1, 1, 1)
    plt.imshow(img[:, :, 0], interpolation='none', cmap=plt.get_cmap('gray'))
    plt.show()

def get_cifar_10_data(base_dir, n_imgs=None, train=True, shuffle=False, log=False, gray=False):
    import pickle
    """
    get cifar 10 data
    """
    def read_one_batch_dict(data_dict: dict):
        """
        given data_dict, return data points and labels
        """
        dps = data_dict[b'data']
        labs = data_dict[b'labels']
        assert len(dps) == len(labs), (len(dps), len(labs))
        dps = np.transpose(dps.reshape([10000, 3, 32, 32]), [0, 2, 3, 1])
        return dps, labs
    
    send_msg(log, 'You are using Cifar10 dataset!', 'warn')
    train_data_batch_fs = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
    test_data_batch_fs = ['test_batch']

    data_batch_fs = train_data_batch_fs if train else test_data_batch_fs

    dps = []
    labs = []
    for batch_f in data_batch_fs:
        with open(path.join(base_dir, batch_f), 'rb') as fo:
            data_dict = pickle.load(fo, encoding='bytes')
            data_dps, data_labs = read_one_batch_dict(data_dict)
            dps.append(data_dps)
            labs.extend(data_labs)
    dps = np.concatenate(dps)  # N, 32, 32, 3
    # FIXME
    # dps = (dps / 255) * 2 - 1
    send_msg(log, "max of data: {}, min of data: {}, mean of data: {}".format(np.max(dps), np.min(dps), np.mean(dps)))

    if gray:
        dps = grayscale(dps)

    labs = np.array(labs)
    return dps, labs


def send_msg(log, msg, level='info'):
    if log:
        assert level in ['info', 'warn', 'error']
        if level == 'info':
            logger.info(msg)
        elif level == 'warn':
            logger.warn(msg)
        else:
            logger.error(msg)
    else:
        print(msg)


def dict_feats_to_np_feats(maybe_dict_feat: (dict, np.ndarray, sparse.csr_matrix)) -> (np.ndarray, sparse.csr_matrix):
    """
    a patch function, since the output of PCANet.transform is dict
    """
    if isinstance(maybe_dict_feat, np.ndarray):
        return maybe_dict_feat
    elif isinstance(maybe_dict_feat, sparse.csr_matrix):
        return maybe_dict_feat
    elif isinstance(maybe_dict_feat, dict):
        X = []
        for i in range(len(maybe_dict_feat)):
            X.append(maybe_dict_feat[i])
    else:
        raise TypeError
    return np.array(X)


def make_tuple(may_be_tuple_or_int: (list, tuple, int)) -> tuple:
    """
    (2, 3) -> (2, 3)
    2 -> (2, 2)
    """
    if isinstance(may_be_tuple_or_int, (list, tuple)):
        return tuple(may_be_tuple_or_int)
    elif isinstance(may_be_tuple_or_int, int):
        return may_be_tuple_or_int, may_be_tuple_or_int
    else:
        raise TypeError("{} is not tuple or int".format(may_be_tuple_or_int))


