# -*- coding: utf-8 -*-
# File: PCANet2.py

import random
from collections import Counter, namedtuple

import gc
from scipy import sparse
import numpy as np
import numba as nb

from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.image import extract_patches_2d

from skimage.util import view_as_blocks

import fourier_utils
import chebyshev_utils
import td_fourier_utils
from basic_utils import randomized_pca, send_msg, dict_feats_to_np_feats, make_tuple


RANDOM_STATE = 233
random.seed(RANDOM_STATE)

Parameter = namedtuple('Parameter', ['l1', 'patch_size1', 'stride1', 'method1', 'fourier_basis_sel1', 
                                     'l2', 'patch_size2', 'stride2', 'method2', 'fourier_basis_sel2',
                                     'reduction_method', 'block_size', 'block_stride', 'feature_method',
                                     'clf'])

clf_dict = {"logistic": LogisticRegression, "svm": LinearSVC}


class BatchData(object):
    def __init__(self, batch_size, dps):
        self._batch_size = batch_size
        self._dps = dps

    def __iter__(self):
        holder = []
        for dp in self._dps:
            holder.append(dp)
            if len(holder) == self._batch_size:
                yield np.array(holder)
                del holder[:]
        if len(holder) > 0:
            yield np.array(holder)

# TODO: add sanity check code, check all dimensions


def pipeline(train_imgs, train_labels, test_imgs, test_labels, param: namedtuple, log=False, rgb=False):
    global RANDOM_STATE
    net = PCANet2(
                 l1=param.l1,
                 patch_size1=param.patch_size1,
                 stride1=param.stride1,
                 method1=param.method1,
                 fourier_basis_sel1=param.fourier_basis_sel1,

                 l2=param.l2,
                 patch_size2=param.patch_size2,
                 stride2=param.stride2,
                 method2=param.method2,
                 fourier_basis_sel2=param.fourier_basis_sel2,

                 reduction_method=param.reduction_method,             
                 block_size=param.block_size,
                 block_stride=param.block_stride,
                 feature_method=param.feature_method,

                 log=log,
                 rgb=rgb
    )
    train_feats = net.fit_transform(train_imgs, train_labels)
    test_feats = net.transform(test_imgs)
    train_feats, test_feats = dict_feats_to_np_feats(train_feats), dict_feats_to_np_feats(test_feats)

    scaler = StandardScaler(with_std=False)
    zero_mean_train_feats = scaler.fit_transform(train_feats)
    X_train = zero_mean_train_feats
    y_train = train_labels

    zero_mean_test_feats = scaler.fit_transform(test_feats)
    X_test = zero_mean_test_feats
    y_test = test_labels
    # X_train = train_feats
    # y_train = train_labels
    # X_test = test_feats
    # y_test = test_labels

    send_msg(log, "training a {} clf ...".format(param.clf))
    CLF = clf_dict.get(param.clf, LinearSVC)
    clf = CLF(
        random_state=RANDOM_STATE,
        max_iter=1000
    )
    # X_train, X_test = sparse.csr_matrix(X_train), sparse.csr_matrix(X_test)

    send_msg(log, "train_feats.shape: {}, test_feats.shape: {}".format(X_train.shape, X_test.shape))
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    send_msg(log, net)
    send_msg(log, 'Mean Accuracy Score: {}'.format(accuracy))
    return accuracy, net, clf


def get_w(X, l, patch_size, method='PCA', fourier_basis_sel='variance', labels=None, rgb=False):
    """
    X: mean removed data output from process_images (n_features x n_samples)
    l: number of filters
    """
    k1, k2 = patch_size
    if method == 'PCA':
        pca = PCA(n_components=l)
        pca.fit(np.transpose(X))
        w = [v.reshape(k1, k2, -1) for v in pca.components_]
    elif method == 'IncrementalPCA':
        pca = IncrementalPCA(n_components=l, copy=False)
        pca.fit(np.transpose(X))
        w = [v.reshape(k1, k2, -1) for v in pca.components_]
    elif method == 'RandomizedPCA':
        w = randomized_pca(X.T, l)
        w = [v.reshape(k1, k2, -1) for v in w]
    elif method == 'Random':
        # TODO: not suitable for cifar 10 experiments
        mu, sigma = 0, 1
        w = []
        for _ in range(l):
            s = np.random.normal(mu, sigma, size=k1*k2).reshape(k1, k2)
            w.append(s)
    elif method == 'Fourier':
        # wrap in the get_w_fourier
        w = fourier_utils.get_w_fourier(X, l, labels, patch_size, fourier_basis_sel, rgb=rgb)
    elif method == '2D_Fourier':
        # wrap in the get_w_fourier
        w = td_fourier_utils.get_w_2d_fourier(X, l, labels, patch_size, fourier_basis_sel, rgb=rgb)
    elif method == 'Chebyshev':
        # TODO: not suitable for cifar 10 experiments
        w = chebyshev_utils.get_w_chebyshev(X, l, labels, patch_size, fourier_basis_sel)
    else:
        raise NotImplementedError("method {} is not supported".format(method))
    w = [np.squeeze(item) for item in w]
    return w


def vectorize_one_img(image, patch_size, stride=1):
    """
    Takes an 2D image, collect the overlapping patches, vectorize them.
    Padding is not considered.
    :image: image should be 2D numpy array (support single channel images only(MNIST))
    :patch_size: a tuple (k1, k2) meaning we take k1*k2 patch
    :stride: int

    return: a k1k2 * m matrix, where m is the number patches collected from the image.
    """
    n_row, n_col = image.shape
    k1, k2 = patch_size
    patches = []
    for row in range(0, n_row - k1 + 1, stride):
        col_patch = [image[row:row + k1, col:col + k2] - np.mean(image[row:row + k1, col:col + k2])\
                        for col in range(0, n_col - k2 + 1, stride)]
        patches.extend(col_patch)

    return np.asarray(patches)


def extract_patches_ong_img(image, patch_size, stride=1):
    assert stride == 1, "sklearn extract_patches_2d only support stride 1"
    patches = extract_patches_2d(image, patch_size)  # n * h * w / n * h * w * 3
    patches_mean = np.mean(patches, axis=(1, 2), keepdims=True)  # n, 1, 1 / n, 1, 1, 3
    patches = patches - patches_mean  # broadcastable
    return patches  # n_patches * h * w / n_patches * h * w * 3


def process_images(images: (list, np.ndarray), patch_size, stride):
    """
    Takes a data loader (batch_size=1) apply vectorize_one_img for each image.
    """
    demo_patches = extract_patches_ong_img(images[0], patch_size=patch_size, stride=stride)
    n_patches = demo_patches.shape[0]
    X = np.zeros([len(images) * n_patches, *demo_patches.shape[1:]])
    del demo_patches
    for i, one_image in enumerate(images):
        X[i * n_patches: i * n_patches + n_patches] = extract_patches_ong_img(one_image, patch_size=patch_size, stride=stride)

    X = X.reshape(X.shape[0], -1)
    return X.T  # n_features * n_samples


def conv_layer(one_image, filters, complex=False):
    """
    """
    assert one_image.ndim in [2, 3], one_image.shape
    h, w, *_ = one_image.shape
    # first padding
    block_h, block_w, *_ = filters[0].shape
    padding_h, padding_w = (block_h - 1) // 2, (block_w - 1) // 2
    if one_image.ndim == 2:  # h, w, gray-scale image
        one_image = np.pad(one_image, [[padding_h, padding_h], [padding_w, padding_w]], mode='constant')
        # already move mean, n,fh,fw
        patches = extract_patches_ong_img(one_image, patch_size=(block_h, block_w), stride=1)
        I = np.empty([len(filters), h, w], dtype=float)
        for idx, filter in enumerate(filters):
            # one_w shape: fh,fw
            filter = filter.reshape(1, *filter.shape)  # -> 1,fh,fw, make it broadcastable
            conved_flatten_img = np.sum(patches * filter, axis=(1, 2))  # -> n_patches,
            conved_img = conved_flatten_img.reshape(h, w)  # n_patches, -> h,w
            I[idx] = conved_img if not complex else np.absolute(conved_img)
    else:  # rgb image
        one_image = np.pad(one_image, [[padding_h, padding_h], [padding_w, padding_w], [0, 0]], mode='constant')
        # already move mean, n,fh,fw,3
        patches = extract_patches_ong_img(one_image, patch_size=(block_h, block_w), stride=1)
        I = np.empty([len(filters), h, w], dtype=float)
        for idx, filter in enumerate(filters):
            # one_w shape: fh,fw,3
            filter = filter.reshape(1, *filter.shape)  # -> 1,fh,fw,3, make it broadcastable
            conved_flatten_img = np.sum(patches * filter, axis=(1, 2, 3))  # -> n_patches,
            conved_img = conved_flatten_img.reshape(h, w)  # n_patches, -> h,w
            I[idx] = conved_img if not complex else absolute(conved_img)
    return I


def binarize(mat: np.ndarray) -> np.ndarray:
    return (mat > 0).astype(int)


class PCANet2(object):
    """
    sklearn-like API for PCANet2: <https://arxiv.org/abs/1404.3606>
    """
    def __init__(self,
                 l1: int,
                 patch_size1: (tuple, int),
                 stride1: int,
                 method1: str,
                 fourier_basis_sel1: str,
                 l2: int,
                 patch_size2: (tuple, int),
                 stride2: int,
                 method2: str,
                 fourier_basis_sel2: str,

                 reduction_method: str,             
                 block_size: (tuple, int),
                 block_stride: int,
                 feature_method: str,

                 log: bool,
                 rgb: bool
                 ):
        self._complex=False # TODO: pass in as an argument

        self._l1 = l1
        self._patch_size1 = make_tuple(patch_size1)
        self._stride1 = stride1
        self._method1 = method1
        self._fourier_basis_sel1 = fourier_basis_sel1
        self._filters1 = None

        self._l2 = l2
        self._patch_size2 = make_tuple(patch_size2)
        self._stride2 = stride2
        self._method2 = method2
        self._fourier_basis_sel2 = fourier_basis_sel2
        self._filters2 = None

        self._reduction_method = reduction_method
        self._block_size = make_tuple(block_size)
        self._block_stride = block_stride
        self._X = None
        self._feature_method = feature_method
        self._levels = [1, 2, 4]
        self._overlapping_ratio = 0

        self._rgb = rgb

        self._feature_reduce_pca = None

        assert feature_method in ['histogram', 'spp_histogram', 'pool_histogram'], 'only support histogram currently'
        assert reduction_method == 'exponent', 'only support exponent currently'

        self._log = log

    def __str__(self):
        msg = "\nPCANet2\n"
        msg += "stage1:\n"
        msg += "\tl1: {}\n".format(self._l1)
        msg += "\tpatch_size1: {}\n".format(self._patch_size1)
        msg += "\tstride1: {}\n".format(self._stride1)
        msg += "\tmethod1: {}\n".format(self._method1)
        msg += "\tfourier_basis_sel1: {}\n".format(self._fourier_basis_sel1)
        
        msg += "stage2:\n"
        msg += "\tl2: {}\n".format(self._l2)
        msg += "\tpatch_size2: {}\n".format(self._patch_size2)
        msg += "\tstride2: {}\n".format(self._stride2)
        msg += "\tmethod2: {}\n".format(self._method2)
        msg += "\tfourier_basis_sel2: {}\n".format(self._fourier_basis_sel2)

        msg += "output stage:\n"
        msg += "\treduction_method: {}\n".format(self._reduction_method)
        msg += "\tblock_size: {}\n".format(self._block_size)
        msg += "\tblock_stride: {}\n".format(self._block_stride)
        msg += "\tfeature_method: {}\n".format(self._feature_method)
        msg += "\tlevels: {}\n".format(self._levels)
        msg += "\toverlapping_ratio: {}\n".format(self._overlapping_ratio)

        return msg

    def fit(self, imgs: (list, np.ndarray), labels: (list, np.ndarray)):
        assert len(imgs) == len(labels), "len(imgs) = {} while len(labels) = {}".format(len(imgs), len(labels))
        send_msg(self._log, "begin fitting")
        '''
        stage 1
        '''
        send_msg(self._log, '\tstage-1')
        send_msg(self._log, '\tprocessing images...')


        X = process_images(imgs, self._patch_size1, self._stride1)  # 提取X

        send_msg(self._log, "\tX.shape: {}".format(X.shape))
        # construct stage 1 filters
        send_msg(self._log, "\tconstructing filters 1")

        w_l1 = get_w(X, self._l1, patch_size=self._patch_size1, method=self._method1,
                     fourier_basis_sel=self._fourier_basis_sel1, labels=labels, rgb=self._rgb)
        del X
        self._filters1 = w_l1
        send_msg(self._log, '\tfilter1.shape: {}'.format(np.array(self._filters1).shape))

        '''
        stage 2
        '''
        send_msg(self._log, '\tstage-2')
        send_msg(self._log, "\tconstruct conved images...")
        conved_images = []
        for img in imgs:
            conved_img = conv_layer(
                img,  # no need to pad by hand, scipy.ndimage.convolve do pad for us
                self._filters1, complex=self._complex)  # a list, len=len(filters)
            conved_images.extend(conved_img)
        send_msg(self._log, "\tlength of conved images: {}, each element's shape: {}".format(len(conved_images), conved_images[0].shape))

        # another pca stage
        send_msg(self._log, '\tprocessing images...')
        Y = process_images(conved_images, patch_size=self._patch_size2, stride=self._stride2)
        send_msg(self._log, "\tY.shape: {}".format(Y.shape))
        send_msg(self._log, "\tconstructing filters 2")
        # construct stage 2 filters
        w_l2 = get_w(Y, self._l2, patch_size=self._patch_size2, method=self._method2,
                     fourier_basis_sel=self._fourier_basis_sel2, labels=labels)
        del imgs
        del conved_images
        del Y
        self._filters2 = w_l2
        send_msg(self._log, '\tfilter2.shape: {}'.format(np.array(self._filters2).shape))
        send_msg(self._log, "done")
        gc.collect()

    def transform(self, raw_images) -> (np.ndarray, sparse.csr_matrix):
        send_msg(self._log, "raw_images.shape: {}".format(raw_images.shape))
        send_msg(self._log, "begin transforming")        
        if self._filters1 is None or self._filters2 is None:
            send_msg(self._log, "return a empty feats, since the model is not fitted yet")
            return []
        # convolution, every img corresponds to l1l2 feature maps
        O = []  # N x L1 x L2 x h x w
        send_msg(self._log, "\tconv stage")
        for i, one_image in enumerate(raw_images):  # i corresponds to img idx
            # stage 1 conv
            I_i = conv_layer(one_image, self._filters1, complex=self._complex)  # L1 x h x w, each img generates L1 stage-1 feature maps
            O_i = []  # L1 x L2 x h x w
            assert len(I_i) == self._l1, len(I_i)
            for l in range(self._l1):
                stage1_feature_map_l = I_i[l]
                O_i_l = conv_layer(stage1_feature_map_l, self._filters2, complex=self._complex)  # L2 x h x w, each stage-1 map generates L2 stage-2 feature maps
                O_i.append(O_i_l)
            O.append(O_i)
        O = np.array(O)
        # send_msg(self._log, "\tO.shape: {}".format(np.array(O).shape))

        # # FIXME: test code, pool before redecution
        # O = block_reduce(O, block_size=(1, 1, 1, 2, 2), func=np.max)
        # send_msg(self._log, "\tafter max pooling: O.shape: {}".format(np.array(O).shape))

        send_msg(self._log, "\treduction stage")    
        T = []  # N x L1 x h x w
        for i in range(len(raw_images)):
            O_i = O[i]  # L1 x L2 x h x w
            # quantalization phase
            O_i = np.array(O_i)
            O_i = np.where(O_i > 0, np.ones_like(O_i), np.zeros_like(O_i))  # binarize

            # reduction phase   
            O_i_0 = O_i[0]  # L2 x h x w             
            l2, h, w = O_i_0.shape
            assert l2 == self._l2

            T_i = []  # L1 x h x w
            for l in range(self._l1):
                O_i_l = O_i[l]  # L2 x h x w
                exponent_mat = (2 ** np.arange(0, self._l2, 1)).reshape([-1, 1, 1])  # L2 x 1 x 1
                integer_image_l = np.sum(
                    exponent_mat * O_i_l,
                    axis=0
                ).astype(int)  # h x w
                T_i.append(integer_image_l)
            T.append(T_i)
        T = np.array(T)
        send_msg(self._log, "\tT.shape: {}".format(np.array(T).shape))  # N x L1 x h x w
        
        send_msg(self._log, "\tfeature stage")

        h, w = T.shape[2:4]
        # FIXME: only for cifar method
        if self._feature_method == 'spp_histogram':
            n_pooled_feature = sum([item ** 2 for item in self._levels])
            feats = np.zeros([T.shape[0], n_pooled_feature * self._l1 * (2 ** self._l2)])

            # pre-compute B
            B = int((1 + (h - self._block_size[0]) / self._block_stride) * (1 + (w - self._block_size[1]) / self._block_stride))

            send_msg(self._log, "\tnumber blocks in each feature map: {}".format(B), 'warn')

            for i in range(len(raw_images)):
                T_i = T[i]  # L1 x h x w
                T_i_idx = np.zeros_like(T_i, dtype=int)
                hist_i = np.zeros([B * self._l1, (2 ** self._l2)], dtype=int)  # B * L1, 2^L2
                final_hist_i = np.zeros([n_pooled_feature, self._l1 * (2 ** self._l2)])  # n_pooled, (l1 * 2 ^ l2)
                cnt = 0

                # histogram phase
                block_stage_for_spp(self._l1, self._block_size, self._block_stride, self._l2, T_i, hist_i, T_i_idx)
                # print(view_as_blocks(T_i_idx[1], (4, 4)))

                for level in self._levels:  # pyramid levels
                    spatial_bin_size = (int(h // level), int(w // level))
                    for bin_idx1 in range(level):
                        for bin_idx2 in range(level):
                            # corresponding blocks idx of this bin
                            valid_T_i_idx = list(set(T_i_idx[:,
                            bin_idx1 * spatial_bin_size[1]: (bin_idx1 + 1) * spatial_bin_size[1],
                            bin_idx2 * spatial_bin_size[0]: (bin_idx2 + 1) * spatial_bin_size[0]].reshape([-1])))

                            # fetch all local histogram corresponding to this bin

                            valid_hist_i = hist_i[valid_T_i_idx, :].reshape([-1, self._l1 * 2 ** self._l2])
                            valid_hist_i = np.amax(valid_hist_i, axis=0)  # l1 * 2^l2
                            final_hist_i[cnt] = valid_hist_i
                            cnt += 1
                final_hist_i = final_hist_i.reshape([-1])  # n_pooled * l1 * 2^l2
                # feats[i] = final_hist_i / np.linalg.norm(final_hist_i, ord=2)
                feats[i] = final_hist_i
        else:
            n_img, _, h, w, *_ = T.shape
            B = int((1 + (h - self._block_size[0]) / self._block_stride) * (1 + (w - self._block_size[1]) / self._block_stride))
            send_msg(self._log, "\tnumber blocks in each feature map: {}".format(B), 'warn')

            feats = np.empty([n_img, B * self._l1 * 2 ** self._l2])  # n, (B * L1 * 2^L2)
            for i in range(len(raw_images)):
                T_i = T[i]  # L1 x h x w
                # histogram phase
                hist_i = np.zeros([B * self._l1 * 2 ** self._l2], dtype=int)  # 2^L2 x L1 x B
                # hist_i = np.zeros([2 ** self._l2 * self._l1 * B], dtype=int)
                if self._feature_method == 'histogram':
                    block_stage(self._l1, self._block_size, self._block_stride, self._l2, T_i, hist_i)
                else:  # pool_histogram (stage-2 pool)
                    pooled_T_i = np.zeros([self._l1, h//2, w//2], dtype=int)
                    pool_stage(self._l1, T_i, pooled_T_i)
                    block_stage(self._l1, self._block_size, self._block_stride, self._l2, pooled_T_i, hist_i)

                feats[i] = hist_i
        send_msg(self._log, "done")        
              
        return np.array(feats)  # N x 2^L2 x L1 x B

    def fit_transform(self, imgs: (list, np.ndarray), labels: (list, np.ndarray)) -> (np.ndarray, sparse.csr_matrix):
        """
        fit and transform
        """
        self.fit(imgs, labels)
        feats = self.transform(imgs)
        return feats


@nb.jit(nopython=True)  # FIXME: may be faster, not profile yet
def block_stage_for_spp(l1, block_size, block_stride, l2, T_i, hist_i, T_i_idx):
    # T_i: L1 x h x w
    overall_counter = 0
    for i in range(0, T_i.shape[1] - block_size[0] + 1, block_stride):
        for j in range(0, T_i.shape[2] - block_size[1] + 1, block_stride):
            for layer in range(l1):
                curr_block = T_i[layer, i:i + block_size[0], j:j + block_size[1]]
                vec_bhist = [0] * (2 ** l2)  # 2^L2
                # np.zeros([2 ** l2], dtype=np.int64)  # 2^L2
                for k in range(block_size[0]):
                    for p in range(block_size[1]):
                        vec_bhist[curr_block[k, p]] += 1
                hist_i[overall_counter, :] = vec_bhist
                T_i_idx[layer, i:i + block_size[0], j:j + block_size[1]] = overall_counter

                overall_counter += 1


@nb.jit(nopython=True)  # FIXME: may be faster, not profile yet
def block_stage(l1, block_size, block_stride, l2, T_i, hist_i):
    # T_i: L1 x h x w
    cnt = 0
    for l in range(l1):
        T_i_l = T_i[l]  # h x w
        for i in range(0, T_i_l.shape[0] - block_size[0] + 1, block_stride):
            for j in range(0, T_i_l.shape[1] - block_size[1] + 1, block_stride):
                curr_block = T_i_l[i:i + block_size[0], j:j + block_size[1]]
                vec_bhist = [0] * (2 ** l2)  # 2^L2
                # np.zeros([2 ** l2], dtype=np.int64)  # 2^L2
                for k in range(block_size[0]):
                    for p in range(block_size[1]):
                        vec_bhist[curr_block[k, p]] += 1
                hist_i[cnt * 2 ** l2: (cnt + 1) * 2 ** l2] = vec_bhist
                cnt += 1


def normal_spp_block_stage(h, w, l1, levels, l2, T_i, overlapping_ratio=0):
    block_hist_vec_list = []
    
    for level in levels:
        # T_i: L1 x h x w
        block_size = (h // level, w // level)
        blocks = view_as_blocks(T_i, (1, *block_size))
        for i in range(level):
            for j in range(level):
                level_block_vec_list = []
                for l in range(l1):
                        block = blocks[l, i, j]
                        block_hist_vec = np.bincount(block.reshape([-1]), minlength=2 ** l2)
                        level_block_vec_list.extend(block_hist_vec)
                        # yield np.array(level_block_vec_list)
                block_hist_vec_list.append(level_block_vec_list)
    block_hist_mat = np.array(block_hist_vec_list)

    return block_hist_mat  # 21 * (L1 * 2^L2)


@nb.jit(nopython=True)  # FIXME: may be faster, not profile yet
def pool_stage(l1, T_i, pooled_T_i):
    # T_i: L1 x h x w
    for l in range(l1):
        T_i_l = T_i[l]  # h x w
        # max pooling first 2*2 stride 2
        for i in range(0, T_i_l.shape[0] - 2 + 1, 2):
            for j in range(0, T_i_l.shape[1] - 2 + 1, 2):
                curr_block = T_i_l[i:i + 2, j:j + 2]
                max_val = int(np.max(curr_block))
                pooled_T_i[l, i//2, j//2] = max_val      



# below is the previous function

# def block_stage(l1, block_size, block_stride, l2, T_i, hist_i):
#     # T_i: L1 x h x w
#     for l in range(l1):
#         T_i_l = T_i[l]  # h x w
#         for i in range(0, T_i_l.shape[0] - block_size[0], block_stride):
#             for j in range(0, T_i_l.shape[1] - block_size[1], block_stride):
#                 curr_block = T_i_l[i:i + block_size[0], j:j + block_size[1]]
#                 block_hist = Counter(curr_block.reshape((curr_block.size,)))
#                 # block_hist = dict(counter)
#                 vec_bhist = np.zeros([2 ** l2])  # 2^L2
#                 # send_msg(block_hist)
#                 for hist_bin, val in block_hist.items():
#                     vec_bhist[hist_bin] = val
#                 hist_i.extend(vec_bhist)
