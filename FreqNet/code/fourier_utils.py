# -*- coding: utf-8 -*-
# File: fourier_utils.py

import numpy as np
import math
import time
from scipy import stats
from scipy.special import softmax
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from collections import Counter

def construct_fourier_basis(N: int, complex: bool=False):
    # w_all contains all the fourier basis for N-dim space
    one_period = 2 * np.pi / N
    w_all = []

    if not complex:
        for k in range((math.floor(N / 2) + 1)):
            one_filter = np.cos([k * one_period * j for j in range(N)]) / math.sqrt(N)
            w_all.append(one_filter)
        for k in range(-1, -math.floor((N - 1) / 2) - 1, -1):
            one_filter = np.sin([k * one_period * j for j in range(N)]) / math.sqrt(N)
            w_all.append(one_filter)
    else:
        for k in range(0, math.floor(N/2)+1):
            one_filter = [np.complex(np.cos(k*one_period*j), -np.sin(k*one_period*j))/math.sqrt(N) for j in range(N)]
            w_all.append(one_filter)

    return w_all


def compute_mean(w_stats_list):
    # 计算组间方差，首先需要计算均值
    w_stats_mean_list = []
    for one_w_unit, one_w_prod_list in w_stats_list:
        cur_tup = (one_w_unit, [-1] * 10)
        for label, one_w_prod in enumerate(one_w_prod_list):
            cur_mean = np.mean(one_w_prod)
            cur_tup[1][label] = cur_mean
        w_stats_mean_list.append(cur_tup)
    print("w_stats_mean_list construction done")
    return w_stats_mean_list


def compute_inter_std_with_mean(w_stats_mean_list):
    w_stats_inter_std_list = []
    for one_w_unit, mean_list in w_stats_mean_list:
        inter_std = np.std(mean_list)
        w_stats_inter_std_list.append((one_w_unit, inter_std))
    print("w_stats_inter_std_list construction done")

    return w_stats_inter_std_list


def compute_intra_std(w_stats_list):
    w_stats_std_list = []
    for one_w_unit, one_w_prod_list in w_stats_list:
        cur_tup = (one_w_unit, [-1] * 10)  # pre-allocate
        for label, one_w_prod in enumerate(one_w_prod_list):
            cur_std = np.std(one_w_prod)
            cur_tup[1][label] = cur_std
        w_stats_std_list.append(cur_tup)
    print("w_stats_intra_std_list construction done")
    return w_stats_std_list


def compute_intra_mag(w_stats_list):
    w_stats_mag_list = []
    for one_w_unit, one_w_prod_list in w_stats_list:
        cur_tup = (one_w_unit, [-1] * 10)  # pre-allocate
        for label, one_w_prod in enumerate(one_w_prod_list):
            cur_mag = np.mean(np.abs(one_w_prod))
            cur_tup[1][label] = cur_mag
        w_stats_mag_list.append(cur_tup)
    print("w_stats_intra_mag_list construction done")
    return w_stats_mag_list


def compute_inter_std(w_stats_list):
    # 计算组间方差，首先需要计算均值
    w_stats_mean_list = compute_mean(w_stats_list)
    # 计算组间方差
    w_stats_inter_std_list = compute_inter_std_with_mean(w_stats_mean_list)
    return w_stats_inter_std_list


def get_w_fourier_group_method(X, l, labels, patch_size, fourier_basis_sel, w_all):
    assert labels is not None
    labels = labels.astype('uint8')
    k1, k2 = patch_size
    N = k1 * k2
    # split the dataset according to its label
    n_imgs = len(labels)
    _, n_samples = X.shape
    samples_per_img = n_samples / n_imgs
    assert round(samples_per_img) - samples_per_img < 1e-4
    samples_per_img = int(samples_per_img)

    # construct [X_0, ..., X_9] list
    label_X_list = [[] for _ in range(10)]
    for idx, label in enumerate(labels):
        label_X_list[label].append(X[:, idx * samples_per_img:(idx + 1) * samples_per_img])
    for label in range(len(label_X_list)):
        label_X_list[label] = np.concatenate(label_X_list[label], axis=1)

    # calculate one_w_prod associated with each one_w_unit and label_X
    start = time.time()
    w_stats_list = []  # list of tuple, (one_w_unit, [one_w_prod_lab0,...,one_w_prod_lab9])
    for one_w in w_all:
        one_w_unit = one_w / np.linalg.norm(one_w)
        cur_tup = (one_w_unit, [-1]*10)
        for label, label_X in enumerate(label_X_list):
            one_w_prod = np.matmul(np.asarray(one_w_unit).reshape((1, N)), label_X)
            cur_tup[1][label] = one_w_prod
        w_stats_list.append(cur_tup)
    end = time.time()
    print("w_stats_list construction done, time elapsed: {}s".format(end - start))

    if fourier_basis_sel == 'group_inter_variance':
        """
        every w_unit 对应 10 (w_unit @ label_X = w_prod), 计算每一个 label 对应的 w_prod 的 mean
        then 每一个 w_unit 对应 10 value, 求方差
        then 每一个 w_unit 对应 1 metric value
        then sort
        """
        # 计算组间方差，首先需要计算均值
        w_stats_mean_list = compute_mean(w_stats_list)
        # 计算组间方差
        w_stats_inter_std_list = compute_inter_std_with_mean(w_stats_mean_list)
        # sort
        w_stats_inter_std_list = sorted(w_stats_inter_std_list, key=lambda tup: tup[1], reverse=True)
        w = [np.asarray(tup[0]).reshape(k1, k2, -1) for tup in w_stats_inter_std_list[:l]]
    elif fourier_basis_sel == 'group_top_mag_union':
        """
        pick top 2 activated frequency for each group(label), then get the union of these w
        """
        w_stats_mean_list = compute_mean(w_stats_list)
        top2_w = dict()
        for group in range(10):
            w_stats_mag = sorted(w_stats_mean_list, key=lambda tup: tup[1][group], reverse=True)
            top2_w[group] = [w_stats_mag[0][0], w_stats_mag[1][0]]
        
        w = []
        for group in top2_w:
            for w_cand in top2_w[group]:
                sel = True
                for one_w in w:
                    if np.linalg.norm(w_cand-one_w, ord=None) < 1e-8:
                        sel = False
                if sel:
                    w.append(w_cand)
        w = [np.asarray(one_w).reshape(k1, k2, -1) for one_w in w]
        print('selected # of w:', len(w))
        
    elif fourier_basis_sel == 'group_intra_inter_var_ratio':
        """
        every w_unit 对应 10 (w_unit @ label_X = w_prod), 计算每一个 label 对应的 w_prod 的 mean, std
        then 每一个 w_unit 对应 10 mean, 求组间方差 and 每一个 w_unit 对应 10 std, 求组内总方差(取平均)
        then 组间 / 组内，每一个 w_unit 对应 1 metric value     
        then sort
        """
        w_stat_intra_std_list = compute_intra_std(w_stats_list)
        w_stat_inter_std_list = compute_inter_std(w_stats_list)
        w_stat_inter_intra_ratio_std_list = []
        for idx in range(len(w_stat_intra_std_list)):
            w_stat_intra_std = w_stat_intra_std_list[idx]
            w_stat_inter_std = w_stat_inter_std_list[idx]
            assert np.all(w_stat_intra_std[0] == w_stat_inter_std[0])
            w_stat_inter_intra_ratio_std_list.append((w_stat_intra_std[0], w_stat_inter_std[1]/np.mean(w_stat_intra_std[1])))
        # sort
        w_stat_inter_intra_ratio_std_list = sorted(w_stat_inter_intra_ratio_std_list, key=lambda tup: tup[1], reverse=True)
        w = [np.asarray(tup[0]).reshape(k1, k2, -1) for tup in w_stat_inter_intra_ratio_std_list[:l]]

    elif fourier_basis_sel == 'group_intra_inter_var_multi':
        """
        every w_unit 对应 10 (w_unit @ label_X = w_prod), 计算每一个 label 对应的 w_prod 的 mean, std
        then 每一个 w_unit 对应 10 mean, 求组间方差 and 每一个 w_unit 对应 10 std, 求组内总方差(取平均)
        then 组间 x 组内，每一个 w_unit 对应 1 metric value     
        then sort
        """
        w_stat_intra_std_list = compute_intra_std(w_stats_list)
        w_stat_inter_std_list = compute_inter_std(w_stats_list)
        w_stat_inter_intra_ratio_std_list = []
        for idx in range(len(w_stat_intra_std_list)):
            w_stat_intra_std = w_stat_intra_std_list[idx]
            w_stat_inter_std = w_stat_inter_std_list[idx]
            assert np.all(w_stat_intra_std[0] == w_stat_inter_std[0])
            w_stat_inter_intra_ratio_std_list.append((w_stat_intra_std[0], w_stat_inter_std[1] * np.mean(w_stat_intra_std[1])))
        # sort
        w_stat_inter_intra_ratio_std_list = sorted(w_stat_inter_intra_ratio_std_list, key=lambda tup: tup[1], reverse=True)
        w = [np.asarray(tup[0]).reshape(k1, k2, -1) for tup in w_stat_inter_intra_ratio_std_list[:l]]
    elif fourier_basis_sel == 'group_intra_var_inter_var':
        """
        every w_unit 对应 10 (w_unit @ label_X = w_prod), 计算每一个 label 对应的 w_prod 的 mean, std
        then 每一个 w_unit 对应 10 std, 再求一次方差
        then 组间 / 组内，每一个 w_unit 对应 1 metric value  
        then sort
        """
        w_stat_intra_std_list = compute_intra_std(w_stats_list)
        w_stat_intra_inter_var_list = []
        for idx in range(len(w_stat_intra_std_list)):
            one_w_unit, label_intra_variance_list = w_stat_intra_std_list[idx]
            w_stat_intra_inter_var_list.append((one_w_unit, np.std(label_intra_variance_list)))
        w_stat_intra_inter_var_list = sorted(w_stat_intra_inter_var_list, key=lambda tup: tup[1], reverse=True)
        w = [np.asarray(tup[0]).reshape(k1, k2, -1) for tup in w_stat_intra_inter_var_list[:l]]

    elif fourier_basis_sel == 'group_intra_mag_inter_var':
        """
        every w_unit 对应 10 (w_unit @ label_X = w_prod), 计算每一个 label 对应的 w_prod 的 magnitude
        then 每一个 w_unit 对应 10 magnitude, 再求一次方差
        then 组间 / 组内，每一个 w_unit 对应 1 metric value  
        then sort
        """
        w_stat_intra_mag_list = compute_intra_mag(w_stats_list)
        w_stat_intra_inter_var_list = []
        for idx in range(len(w_stat_intra_mag_list)):
            one_w_unit, label_intra_mag_list = w_stat_intra_mag_list[idx]
            w_stat_intra_inter_var_list.append((one_w_unit, np.std(label_intra_mag_list)))
        w_stat_intra_inter_var_list = sorted(w_stat_intra_inter_var_list, key=lambda tup: tup[1], reverse=True)
        w = [np.asarray(tup[0]).reshape(k1, k2, -1) for tup in w_stat_intra_inter_var_list[:l]]

    elif fourier_basis_sel == 'group_intra_mag_entropy':
        """
        every w_unit 对应 10 (w_unit @ label_X = w_prod), 计算每一个 label 对应的 w_prod 的 magnitude
        then 每一个 w_unit 对应 10 magnitude, standardize, entropy -> 1 metric value
        then sort
        """
        w_stat_intra_mag_list = compute_intra_mag(w_stats_list)
        w_stat_entropy_list = []
        for idx in range(len(w_stat_intra_mag_list)):
            one_w_unit, label_intra_mag_list = w_stat_intra_mag_list[idx]
            label_intra_mag_prob_list = np.array(label_intra_mag_list) / np.sum(label_intra_mag_list)
            entropy = stats.entropy(label_intra_mag_prob_list)
            w_stat_entropy_list.append((one_w_unit, entropy))
        w_stat_entropy_list = sorted(w_stat_entropy_list, key=lambda tup: tup[1], reverse=True)
        w = [np.asarray(tup[0]).reshape(k1, k2, -1) for tup in w_stat_entropy_list[:l]]
    elif fourier_basis_sel == 'group_intra_mag_entropy_reverse':
        """
        every w_unit 对应 10 (w_unit @ label_X = w_prod), 计算每一个 label 对应的 w_prod 的 magnitude
        then 每一个 w_unit 对应 10 magnitude, standardize, entropy -> 1 metric value
        then sort
        """
        w_stat_intra_mag_list = compute_intra_mag(w_stats_list)
        w_stat_entropy_list = []
        for idx in range(len(w_stat_intra_mag_list)):
            one_w_unit, label_intra_mag_list = w_stat_intra_mag_list[idx]
            label_intra_mag_prob_list = np.array(label_intra_mag_list) / np.sum(label_intra_mag_list)
            entropy = stats.entropy(label_intra_mag_prob_list)
            w_stat_entropy_list.append((one_w_unit, entropy))
        w_stat_entropy_list = sorted(w_stat_entropy_list, key=lambda tup: tup[1], reverse=False)
        w = [np.asarray(tup[0]).reshape(k1, k2, -1) for tup in w_stat_entropy_list[:l]]
    elif fourier_basis_sel == 'group_intra_mag_entropy_softmax':
        """
        every w_unit 对应 10 (w_unit @ label_X = w_prod), 计算每一个 label 对应的 w_prod 的 magnitude
        then 每一个 w_unit 对应 10 magnitude, standardize, entropy -> 1 metric value
        then sort
        """
        w_stat_intra_mag_list = compute_intra_mag(w_stats_list)
        w_stat_entropy_list = []
        for idx in range(len(w_stat_intra_mag_list)):
            one_w_unit, label_intra_mag_list = w_stat_intra_mag_list[idx]
            label_intra_mag_prob_list = softmax(label_intra_mag_list)
            entropy = stats.entropy(label_intra_mag_prob_list)
            w_stat_entropy_list.append((one_w_unit, entropy))
        w_stat_entropy_list = sorted(w_stat_entropy_list, key=lambda tup: tup[1], reverse=True)
        w = [np.asarray(tup[0]).reshape(k1, k2, -1) for tup in w_stat_entropy_list[:l]]
    elif fourier_basis_sel == 'group_intra_mag_entropy_softmax_reverse':
        """
        every w_unit 对应 10 (w_unit @ label_X = w_prod), 计算每一个 label 对应的 w_prod 的 magnitude
        then 每一个 w_unit 对应 10 magnitude, standardize, entropy -> 1 metric value
        then sort
        """
        w_stat_intra_mag_list = compute_intra_mag(w_stats_list)
        w_stat_entropy_list = []
        for idx in range(len(w_stat_intra_mag_list)):
            one_w_unit, label_intra_mag_list = w_stat_intra_mag_list[idx]
            label_intra_mag_prob_list = softmax(label_intra_mag_list)
            entropy = stats.entropy(label_intra_mag_prob_list)
            w_stat_entropy_list.append((one_w_unit, entropy))
        w_stat_entropy_list = sorted(w_stat_entropy_list, key=lambda tup: tup[1], reverse=False)
        w = [np.asarray(tup[0]).reshape(k1, k2, -1) for tup in w_stat_entropy_list[:l]]
    elif fourier_basis_sel == 'group_max_mag_union':
        """
        every w_unit 对应 10 (w_unit @ label_X = w_prod), 计算每一个 label 对应的 w_prod 的 magnitude, 取 top 3 maginitude for every label
        然后 union 所有取到的 w_unit
        then sort
        """
        w_stat_intra_mag_list = compute_intra_mag(w_stats_list)  # list of tuple, (one_w_unit, [mag_lab0, ..., mag_lab9])
        mag_mat = np.array([tup[1] for tup in w_stat_intra_mag_list])  # list of [[mag_lab0, ..., mag_lab9]]  -> l x 10
        w_unit_vec = np.array([tup[0] for tup in w_stat_intra_mag_list])  # l,
        w_unit_vec_ind = np.array(range(len(w_unit_vec)))  # l,
        w_unit_mat_ind = np.tile(np.reshape(w_unit_vec_ind, [-1, 1]), [1, 10])  # l, 10
        argsort_ind = np.argsort(mag_mat, axis=0)
        sorted_w_unit_ind = np.take_along_axis(w_unit_mat_ind, argsort_ind, axis=0)  # l, 10

        # take top l//2 of each label  3, 10
        top_3_w_unit_mat_ind = sorted_w_unit_ind[-int(l//2):, ...]
        print(top_3_w_unit_mat_ind)
        # take union
        ind_list = top_3_w_unit_mat_ind.reshape([-1,]).tolist()
        counter = Counter(ind_list)
        sorted_counter_list = counter.most_common()
        if len(sorted_counter_list) < l:
            print('\t\t WARN: the length of unioned basis list:{} is smaller than {}'.format(len(sorted_counter_list), l))
            
        selected_ind = [tup[0] for tup in sorted_counter_list[:min(len(sorted_counter_list), l)]]  # 
        selected_w_unit = [np.array(w_unit).reshape(k1, k2, -1) for w_unit in w_unit_vec[selected_ind]]
        w = selected_w_unit
    else:
        raise NotImplementedError

    return w


def get_w_fourier(X, l, labels, patch_size, fourier_basis_sel='variance', rgb=False):
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
    w_all = construct_fourier_basis(N)
    
    # w_all contains all the fourier basis for N-dim space,
    # then we try to find "the most important" l base vectors:
    #     1. calculate the inner products of each basis and each patch
    #     2. get abs mean and variance of the inner products
    #     3. pick top l basis (variance? magnitude?)
    if fourier_basis_sel in ['variance', 'magnitude']:
        w_stats = []
        for one_w in w_all:
            one_w_unit = one_w / np.linalg.norm(one_w)
            one_w_prod = np.matmul(np.asarray(one_w_unit).reshape((1, -1)), X)
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


def visualize_fourier_basis(fourier_bases: np.ndarray, l=None, k1=None, k2=None, save_pic=False, pic_name=None):
    """
    Args:
        * fourier_bases: of shape l x k1 x k2 or l x k1 x k2 x 3
        * l: number of filters
        * k1: height
        * k2: width
    """
    assert len(fourier_bases.shape) in [3, 4], 'only accept l x k1 x k2 or l x k1 x k2 x 3 shaped basis, but got {}'.format(fourier_bases.shape)
    if l is None or k1 is None or k2 is None:
        print('\t\t escape sanity check, fourier_bases.shape={}'.format(fourier_bases.shape))
        l, k1, k2 = fourier_bases.shape
    if save_pic:
        assert pic_name is not None, "save_pic with no pic name"
    assert (l, k1, k2) == fourier_bases.shape[:3], 'sanity check failure, {} != {},{},{}'.format(fourier_bases.shape, l, k1, k2)
    fourier_bases = fourier_bases + np.abs(np.min(fourier_bases, axis=(1, 2), keepdims=True))
    fourier_bases = fourier_bases / np.max(fourier_bases, axis=(1, 2), keepdims=True)  # 
    n_row = int(math.ceil(l / 8))
    _, axes = plt.subplots(n_row, 8, sharey=True)
    if len(fourier_bases.shape) == 4:
        color_map = None
    else:
        color_map = 'gray'
    for row in range(n_row):
        for col in range(8):
            try:
                if n_row == 1:
                    axes[col].imshow(fourier_bases[row * 8 + col], cmap=color_map)
                else:
                    axes[row, col].imshow(fourier_bases[row * 8 + col], cmap=color_map)
            except IndexError:
                print(row, col)
                if n_row == 1:
                    axes[col].imshow(np.zeros_like(fourier_bases[0]), cmap=color_map)
                else:
                    axes[row, col].imshow(np.zeros_like(fourier_bases[0]), cmap=color_map)
    if save_pic:
        plt.savefig(pic_name)
    else:
        plt.show()
