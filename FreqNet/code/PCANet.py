# -*- coding: utf-8 -*-
# File: PCANet.py

import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.image import extract_patches_2d
from tqdm import tqdm
from collections import Counter, namedtuple, OrderedDict
import fourier_utils
import wavelet_utils
import chebyshev_utils
import random
from basic_utils import randomized_pca, send_msg, dict_feats_to_np_feats, make_tuple

RANDOM_STATE = 233
random.seed(RANDOM_STATE)


class Parameter(object):
    """
    namedtuple no longer statisfy our requirement
    adding new parameter will be easier and not offend previous json config file
    """
    def __init__(self, l, num_images, patch_size, stride, block_size, block_stride, method='RandomizedPCA', 
                 fourier_basis_sel='', reduction_method='exponent', clf='svm', feature_method='histogram',
                 quantization_method='binarize', wavelet='db1', wavelet_decomp_level=1):

        super(Parameter, self).__init__()
        self.l = l
        self.num_images = num_images
        self.patch_size = patch_size
        self.stride = stride
        self.block_size = block_size
        self.block_stride = block_stride
        self.method = method
        self.fourier_basis_sel = fourier_basis_sel
        self.reduction_method = reduction_method
        self.clf = clf
        self.feature_method = feature_method
        self.quantization_method = quantization_method
        self.wavelet = wavelet
        self.wavelet_decomp_level = wavelet_decomp_level


clf_dict = {"logistic": LogisticRegression, "svm": LinearSVC}


def pipeline(train_imgs, train_labels, test_imgs, test_labels, param: namedtuple, log=False):
    global RANDOM_STATE
    net = PCANet(
        l=param.l,
        num_images=param.num_images,
        patch_size=param.patch_size,
        stride=param.stride,
        block_size=param.block_size,
        block_stride=param.block_stride,
        method=param.method,
        fourier_basis_sel=param.fourier_basis_sel,
        reduction_method=param.reduction_method,
        feature_method=param.feature_method,
        quantization_method=param.quantization_method,
        wavelet=param.wavelet,
        wavelet_decomp_level=param.wavelet_decomp_level,
        log=log
    )
    train_feats = net.fit_transform(train_imgs, train_labels)
    test_feats = net.transform(test_imgs)
    train_feats, test_feats = dict_feats_to_np_feats(train_feats), dict_feats_to_np_feats(test_feats)
    send_msg(log, "train_feats.shape: {}, test_feats.shape: {}".format(train_feats.shape, test_feats.shape))
    scaler = StandardScaler(with_std=False)
    zero_mean_train_feats = scaler.fit_transform(train_feats)
    X_train = zero_mean_train_feats
    y_train = train_labels
    zero_mean_test_feats = scaler.fit_transform(test_feats)
    X_test = zero_mean_test_feats
    y_test = test_labels
    CLF = clf_dict.get(param.clf, LinearSVC)
    clf = CLF(
        random_state=RANDOM_STATE,
        max_iter=2000
    )
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    send_msg(log, net)
    send_msg(log, 'Mean Accuracy Score: {}'.format(accuracy))
    return accuracy, net, clf


def vectorize_one_img(image, patch_size, stride=1, padding=None):
    """
    Takes an 2D image, collect the overlapping patches, vectorize them.
    Padding is not considered.
    :image: image should be 2D numpy array (support single channel images only(MNIST))
    :patch_size: a tuple (k1, k2) meaning we take k1*k2 patch
    :stride: int

    return: a k1k2 * m matrix, where m is the number patches collected from the image.
    """
    if padding:
        raise NotImplementedError
    n_row, n_col = image.shape
    k1, k2 = patch_size
    patches = []
    mean_patches = []
    for row in range(0, n_row - k1 + 1, stride):
        for col in range(0, n_col - k2 + 1, stride):
            one_patch = np.copy(image[row:row + k1, col:col + k2]).reshape((k1 * k2,))
            # remove mean
            one_patch = one_patch - np.mean(one_patch)
            mean_patches.append(np.mean(one_patch))
            patches.append(one_patch)

    return np.transpose(np.asarray(patches)), np.asarray(mean_patches)


def get_w(X, l, patch_size, method='PCA', fourier_basis_sel='variance', wavelet='db1', wavelet_decomp_level=1, labels=None):
    """
    X: mean removed data output from process_images (n_features x n_samples)
    l: number of filters
    """
    k1, k2 = patch_size
    if method == 'PCA':
        pca = PCA(n_components=l)
        pca.fit(np.transpose(X))
        w = [v.reshape(k1, k2) for v in pca.components_]
    if method == 'IncrementalPCA':
        pca = IncrementalPCA(n_components=l)
        pca.fit(np.transpose(X))
        w = [v.reshape(k1, k2) for v in pca.components_]
    elif method == 'RandomizedPCA':
        w = randomized_pca(X.T, l)  # l * k1k2
        w = [v.reshape(k1, k2) for v in w]
    elif method == 'Random':
        mu, sigma = 0, 1
        w = []
        for _ in range(l):
            s = np.random.normal(mu, sigma, size=k1*k2).reshape(k1, k2)
            w.append(s)
    elif method == 'Fourier':
        # wrap in the get_w_fourier
        w = fourier_utils.get_w_fourier(X, l, labels, patch_size, fourier_basis_sel)
    elif method == 'Chebyshev':
        w = chebyshev_utils.get_w_chebyshev(X, l, labels, patch_size, fourier_basis_sel)
    elif method == 'Wavelet':
        l_a, l_d = l
        sel_coeffs_idx, conv_res = wavelet_utils.get_coeffs_idx_wavelets(X.T, l_a=l_a, l_d=l_d, wavelet=wavelet, level=wavelet_decomp_level, basis_sel=fourier_basis_sel)
        return sel_coeffs_idx, conv_res
    else:
        raise NotImplementedError("method {} is not supported".format(method))

    return w


def process_images(images: (list, np.ndarray), patch_size, stride):
    """
    Takes a data loader (batch_size=1) apply vectorize_one_img for each image.
    """
    # X = None
    X = []
    for one_image in tqdm(images):
        one_x, _ = vectorize_one_img(one_image, patch_size=patch_size, stride=stride)
        # print('one_x shape:', one_x.shape)
        X.append(one_x)
        # X = one_x if X is None else np.concatenate((X, one_x), axis=1)
    # n_features * n_samples
    X = np.concatenate(X, axis=1)

    return X


def extract_patches_ong_img(image, patch_size, stride=1):
    assert stride == 1, "sklearn extract_patches_2d only support stride 1"
    patches = extract_patches_2d(image, patch_size)  # n * h * w / n * h * w * 3
    patches_mean = np.mean(patches, axis=(1, 2), keepdims=True)  # n, 1, 1 / n, 1, 1, 3
    patches = patches - patches_mean  # broadcastable
    return patches  # n_patches * h * w / n_patches * h * w * 3


def conv_layer(one_image, w, coeffs_idx, **kwargs):
    # I_l -> convolve with lth filter
    if w is not None:
        filters = w
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
                I[idx] = conved_img
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
                I[idx] = conved_img
    else:
        # for wavelet special case
        wavelet = kwargs['wavelet']
        level = kwargs['level']
        patch_size = kwargs['patch_size']
        patches, _ = vectorize_one_img(one_image, patch_size=patch_size, padding=False) # as padding is not added when generating patches, the conved image will shrink size
        
        wave_coefs = wavelet_utils.get_conv_res_wavelets(patches.T, coeffs_idx, wavelet=wavelet, level=level)
        # reconstruct the convoluted images
        wave_coefs = np.transpose(wave_coefs)
        I = [wave_coefs[i].reshape(one_image.shape[0]+1-patch_size[0], one_image.shape[1]+1-patch_size[1]) for i in range(wave_coefs.shape[0])]

    return I


def relu(mat: np.ndarray) -> np.ndarray:
    return np.where(mat > 0, mat, np.zeros_like(mat))


def binarize(mat: np.ndarray) -> np.ndarray:
    return (mat > 0).astype(int)


class PCANet(object):
    """
    sklearn-like API for PCANet: <https://arxiv.org/abs/1404.3606>
    """
    def __init__(self,
                 l: int,
                 num_images: int,
                 patch_size: (tuple, int),
                 stride: int,
                 block_size: (tuple, int),
                 block_stride: int,
                 method: str = 'PCA',
                 fourier_basis_sel: str = 'variance',
                 wavelet: str = 'db1',
                 wavelet_decomp_level: int = 1,
                 wavelet_la: float = 0.5,
                 reduction_method: str = 'exponent',
                 stage: int = 1,
                 l2: int = 2,
                 feature_method: str = 'histogram',
                 quantization_method: str = 'binarize',
                 log: bool = False
                 ):

        self._l = l
        self._wavelet_la = wavelet_la
        self._num_images = num_images
        self._patch_size = make_tuple(patch_size)
        self._stride = stride
        self._block_size = make_tuple(block_size)
        self._block_stride = block_stride
        self._method = method
        self._fourier_basis_sel = fourier_basis_sel
        self._filters = None
        self._filters_2 = None
        self._X = None
        self._stage = stage
        self._l2 = l2
        self._reduction_method = reduction_method
        self._feature_method = feature_method
        self._quantization_method = quantization_method
        self._wavelet = wavelet
        self._wavelet_decomp_level = wavelet_decomp_level
        self._log = log
        self._select_coeffs_idx = None
        assert stage == 1, "use PCANet2.py for PCANet2 pls"

    def __str__(self):
        msg = "PCANet\n"
        msg += "l: {}\n".format(self._l)
        msg += "patch_size: {}\n".format(self._patch_size)
        msg += "stride: {}\n".format(self._stride)
        msg += "block_size: {}\n".format(self._block_size)
        msg += "block_stride: {}\n".format(self._block_stride)
        msg += "method: {}\n".format(self._method)
        msg += "stage: {}\n".format(self._stage)
        msg += "reduction_method: {}\n".format(self._reduction_method)
        msg += "feature_method: {}\n".format(self._feature_method)
        msg += "fourier_basis_sel: {}\n".format(self._fourier_basis_sel)
        return msg

    def fit(self, imgs: (list, np.ndarray), labels: (list, np.ndarray)):
        send_msg(self._log, "begin fitting")
        assert len(imgs) == len(labels), "len(imgs) = {} while len(labels) = {}".format(len(imgs), len(labels))
        send_msg(self._log, 'processing images...')
        X = process_images(imgs, self._patch_size, self._stride)
        send_msg(self._log, "X.shape: {}".format(X.shape))
        # as matrix transform representation is currently not supported, Wavelets based filter
        # get_w will give the selected indices of selected coefficients instead of convolution filters
        if self._method == 'Wavelet':
            send_msg(self._log, "constructing filters")

            l_a = int(self._wavelet_la * self._l)
            l_b = self._l - l_a
            select_coeffs_idx, _ = get_w(X, (l_a, l_b), patch_size=self._patch_size, method=self._method, wavelet=self._wavelet,
                                                 wavelet_decomp_level=self._wavelet_decomp_level, labels=labels)
            self._select_coeffs_idx = select_coeffs_idx
        else:
            send_msg(self._log, "constructing filters")

            w_l1 = get_w(X, self._l, patch_size=self._patch_size, method=self._method,
                         fourier_basis_sel=self._fourier_basis_sel, labels=labels)
            # construct stage 1 filters
            self._filters = w_l1
            send_msg(self._log, '\tfilter.shape: {}'.format(np.array(self._filters).shape))
        self._X = X

    def refit_with_other_method(self, method, fourier_basis_sel):
        """
        no need to calculate X again(call process_images is time-consuming)
        """
        self._method = method
        self._fourier_basis_sel = fourier_basis_sel
        w_l1 = get_w(self._X, self._l, patch_size=self._patch_size, method=self._method,
                     fourier_basis_sel=self._fourier_basis_sel)
        self._filters = w_l1

    def fit_output_pca(self, raw_images, dim=20):
        
        if self._X is None:
            send_msg(self._log, "return a empty dict, since the model is not fitted yet")
            return dict()
        # convolution
        I_layer1 = dict()
        I_freq = dict()
        for i, one_image in enumerate(raw_images):
            I_i = conv_layer(one_image, self._filters)
            I_layer1[i] = I_i
            for j, one_I in enumerate(I_i):
                if not j in I_freq:
                    I_freq[j] = []
                I_freq[j].append(one_I.reshape(one_I.size, ))
        send_msg(self._log, "conv stage finished")

        self._pca_freq = dict()
        for i in I_freq:
            I_freq[i] = np.asarray(I_freq[i])
            # de-mean
            I_freq[i] = I_freq[i] - (I_freq[i].mean(axis=1)).reshape(-1, 1)
            pca = PCA(n_components=dim)
            pca.fit(I_freq[i])
            self._pca_freq[i] = pca
            
    def transform_output_pca(self, raw_images):

        if self._pca_freq is None:
            send_msg(self._log, "return a empty dict, since the model is not fitted yet")
            return dict()
        # convolution
        I_layer1 = dict()
        feats = [-1] * len(raw_images)
        for i, one_image in enumerate(raw_images):
            I_i = conv_layer(one_image, self._filters)
            feats_one_img = []
            for j, one_I in enumerate(I_i):
                reduced_feats = self._pca_freq[j].transform(one_I.reshape(1, -1))
                feats_one_img.extend(reduced_feats)
            feats[i] = np.asarray(feats_one_img).reshape(-1,)
        send_msg(self._log, "conv & output pca stage finished")
        return np.asarray(feats)

    def transform(self, raw_images) -> np.ndarray:
        send_msg(self._log, "begin transforming")        
        if self._X is None:
            send_msg(self._log, "return a empty dict, since the model is not fitted yet")
            return dict()
        send_msg(self._log, "conv stage")
        # convolution
        I_layer1 = []
        kwargs = {}
        kwargs['wavelet'] = self._wavelet
        kwargs['level'] = self._wavelet_decomp_level
        kwargs['patch_size'] = self._patch_size
        for i, one_image in enumerate(raw_images):
            I_i = conv_layer(one_image, self._filters, self._select_coeffs_idx, **kwargs)
            I_layer1.append(I_i)

        send_msg(self._log, "conv stage finished")
        send_msg(self._log, "\tI_layer1.shape: {}".format(np.array(I_layer1).shape))
        
        send_msg(self._log, "reduction stage")    

        # int_img_dict = dict()
        int_img_list = []
        # output layer
        # idx refers to img idx
        for idx in range(len(I_layer1)):
            curr_images = I_layer1[idx]
            integer_image = None
            # quantization method
            if self._quantization_method == 'binarize':
                curr_images = [binarize(one_image) for one_image in curr_images]
            elif self._quantization_method == 'relu':
                curr_images = [relu(one_image) for one_image in curr_images]
            elif self._quantization_method == 'identity':
                pass
            else:
                raise NotImplementedError

            if self._reduction_method == 'concat':  # concat method offer a upper bound
                # int_img_dict[idx] = np.stack(curr_images, 2)
                # int_img_list.append(curr_images)
                int_img_list.append(np.stack(curr_images, 2))

            else:
                # FIXME: for exponent method, reverse's behavior is opposite from others, but resembles original paper
                
                for j, one_image in enumerate(reversed(curr_images)):
                    # reduction method
                    if integer_image is None:
                        if self._reduction_method == 'exponent':
                            integer_image = one_image * (2 ** 0)
                        elif self._reduction_method == 'add':
                            integer_image = one_image * 1
                        elif self._reduction_method == 'linear_add':
                            integer_image = one_image * 1
                        elif self._reduction_method == 'square_add':
                            integer_image = one_image * (1 ** 2)
                        elif self._reduction_method == 'cube_add':
                            integer_image = one_image * (1 ** 3)
                        else:
                            raise NotImplementedError
                    else:
                        if self._reduction_method == 'exponent':
                            integer_image = integer_image * 2 + one_image
                        elif self._reduction_method == 'add':
                            integer_image = integer_image + one_image
                        elif self._reduction_method == 'linear_add':
                            integer_image = integer_image + one_image * (j + 1)
                        elif self._reduction_method == 'square_add':
                            integer_image = integer_image + np.round(one_image * (j + 1) ** 2 / 2).astype(int)
                        elif self._reduction_method == 'cube_add':
                            integer_image = integer_image + np.round(one_image * (j + 1) ** 3 / 3).astype(int)
                        else:
                            raise NotImplementedError

                # int_img_dict[idx] = integer_image
                int_img_list.append(integer_image)
        send_msg(self._log, "reduction stage finished")        
        send_msg(self._log, "\tint_img_list.shape: {}".format(np.array(int_img_list).shape))
        
        send_msg(self._log, "feature stage")    
        # feats = dict()
        # feats = [-1] * len(int_img_list)
        feats = []
        for idx in range(len(int_img_list)):
            # integer_image = int_img_dict[idx]
            integer_image = int_img_list[idx]
            all_vec_bhist = []
            for i in range(0, integer_image.shape[0] - self._block_size[0] + 1, self._block_stride):
                for j in range(0, integer_image.shape[1] - self._block_size[1] + 1, self._block_stride):
                    curr_block = integer_image[i:i + self._block_size[0], j:j + self._block_size[1]]\
                                            .reshape(*self._block_size, -1)

                    # feature method
                    if self._feature_method == 'histogram':
                        counter = Counter(curr_block.reshape((curr_block.size,)))
                        block_hist = dict(counter)
                        # vectorize bhist
                        if self._reduction_method == 'exponent':
                            vec_bhist = np.zeros((1 << self._l,))
                        elif self._reduction_method == 'add':
                            vec_bhist = np.zeros((self._l + 1,))
                        elif self._reduction_method == 'linear_add':
                            vec_bhist = np.zeros([round((self._l * (self._l + 1)) / 2) + 1,])
                        elif self._reduction_method == 'square_add':
                            vec_bhist = np.zeros([round((self._l * (self._l + 1) * (self._l + 0.5)) / 9) + self._l ** 2 + 1, ])
                        elif self._reduction_method == 'cube_add':
                            vec_bhist = np.zeros([int((self._l * (self._l + 1)) / 2) ** 2 + 1, ])

                        for hist_bin, val in block_hist.items():
                            vec_bhist[hist_bin] = val
                        all_vec_bhist.extend(vec_bhist)

                    elif self._feature_method == 'avg_pooling':
                        block_avg = np.mean(curr_block)
                        all_vec_bhist.append(block_avg)
                    elif self._feature_method == 'max_pooling':
                        block_max = np.max(curr_block)
                        all_vec_bhist.append(block_max)
                    elif self._feature_method == 'min_pooling':
                        block_min = np.min(curr_block)
                        all_vec_bhist.append(block_min)
                    elif self._feature_method == 'variance_pooling':
                        block_var = np.var(curr_block)
                        all_vec_bhist.append(block_var)
                    elif self._feature_method == 'max_avg_min_pooling':
                        block_avg = np.mean(curr_block)
                        block_max = np.max(curr_block)
                        block_min = np.min(curr_block)
                        all_vec_bhist.extend([block_avg, block_max, block_min])
                    elif self._feature_method == 'quantile':
                        quantile_list = np.linspace(0, 1, self._l)
                        block_quantile = np.quantile(curr_block, quantile_list).tolist()
                        all_vec_bhist.extend(block_quantile + [np.mean(curr_block), np.var(curr_block)])
                    elif self._feature_method == 'naive_quantile':  # min, max, mean, var
                        for dim in range(curr_block.shape[2]):
                            block_2d = curr_block[..., dim]
                            quantile_list = np.linspace(0, 1, 2)
                            block_quantile = np.quantile(block_2d, quantile_list).tolist()
                            all_vec_bhist.extend(block_quantile + [np.mean(block_2d), np.var(block_2d)])
                    elif self._feature_method == 'exponent_quantile':
                        quantile_list = np.linspace(0, 1, 2 ** self._l)
                        block_quantile = np.quantile(curr_block, quantile_list).tolist()
                        all_vec_bhist.extend(block_quantile + [np.mean(curr_block), np.var(curr_block)])
                    else:
                        raise NotImplementedError
            # feats[idx] = all_vec_bhist
            feats.append(all_vec_bhist)
        
        return np.array(feats)

    def fit_transform(self, imgs: (list, np.ndarray), labels: (list, np.ndarray)) -> dict:
        """
        fit and transform
        """
        self.fit(imgs, labels)
        return self.transform(imgs)

    def get_filters(self):
        filter_len = self._filters[0].size
        v = np.asarray([one_filter.reshape(filter_len,) for one_filter in self._filters])
        return v
