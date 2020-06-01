'''
a python realization of PCA-Net based on "PCANet: A Simple Deep Learning Baseline for Image Classification?"
'''
import numpy as np
from sklearn.decomposition import PCA
from scipy import ndimage
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import math

def binarize(mat):
    return (mat>0).astype(int)

'''
Around each pixel, we take k1*k2 patch, and we collect all (overlapping) patches
to the ith image and then vectorize them, then we subtract patch mean from each patch.
'''
def vectorize_one_img(image, patch_size, stride=1):
    '''
    Takes an 2D image, collect the overlapping patches, vectorize them.
    Padding is not considered.
    :image: image should be 2D numpy array (support single channel images only(MNIST))
    :patch_size: a tuple (k1, k2) meaning we take k1*k2 patch
    :stride: int

    return: a k1k2 * m matrix, where m is the number patches collected from the image. 
    '''
    n_row, n_col = image.shape
    k1, k2 = patch_size
    patches = []
    for row in range(0, n_row-k1, stride):
        for col in range(0, n_col-k2, stride):
            one_patch = np.copy(image[row:row+k1, col:col+k2]).reshape((k1*k2,))
            # remove mean
            one_patch = one_patch - np.mean(one_patch)
            patches.append(one_patch)
    
    return np.transpose(np.asarray(patches))

def process_images(images, patch_size, stride=1):
    '''
    Takes a data loader (batch_size=1) apply vectorize_one_img for each image.
    '''
    print('processing images...')
    X = None
    for one_image in tqdm(images):
        one_x = vectorize_one_img(one_image, patch_size=patch_size, stride=stride) 
        #print('one_x shape:', one_x.shape)
        X = one_x if X is None else np.concatenate((X, one_x), axis=1)
    return X

def get_w(X, l, patch_size, method='PCA', fourier_basis_sel='variance'):
    '''
    X: mean removed data output from process_images (n_features x n_samples)
    l: number of filters
    '''
    print('X:', X)
    k1, k2 = patch_size
    if method == 'PCA':
        pca = PCA(n_components=l)
        pca.fit(np.transpose(X))
        w = [v.reshape(k1, k2) for v in pca.components_]
    elif method == 'Random':
        mu, sigma = 0, 1
        w = []
        for _ in range(l):
            s = np.random.normal(mu, sigma, size=k1*k2).reshape(k1, k2)
            w.append(s) 
    elif method == 'Fourier':
        # N = k1 * k2
        # cosin basis: 
        #  c_k = {cos(2*k*pi/N * 0), ..., cos(2*k*pi/N * (N-1))}/sqrt(N)
        # sin basis:
        #  s_k = {sin(2*k*pi/N * 0), ..., sin(2*k*pi/N * (N-1))}/sqrt(N)
        # where 
        # for c_k, k in {0, 1, ..., trunc(N/2)}
        # for s_k, k in {-trunc((N-1)/2), ..., -1)}
        N = k1 * k2
        one_period = 2 * np.pi / N
        w_all = []
        for k in range((math.floor(N/2)+1)):
            one_filter = np.cos([k * one_period * j for j in range(N)])/math.sqrt(N)
            w_all.append(one_filter)
        for k in range(-1, -math.floor((N-1)/2)-1, -1):
            one_filter = np.sin([k * one_period * j for j in range(N)])/math.sqrt(N)
            w_all.append(one_filter)
        # w_all contains all the fourier basis for N-dim space,
        # then we try to find "the most important" l base vectors:
        #     1. calculate the inner products of each basis and each patch
        #     2. get abs mean and variance of the inner products
        #     3. pick top l basis (variance? magnitude?)
        if fourier_basis_sel in ['variance', 'magnitude']:
            w_stats = []
            for one_w in w_all:
                one_w_unit = one_w / np.linalg.norm(one_w)
                one_w_prod = np.matmul(np.asarray(one_w_unit).reshape((1, N)), X)
                w_stats.append((one_w_unit, np.mean(abs(one_w_prod)), np.std(one_w_prod)))
            if fourier_basis_sel == 'variance':
                # sort w by variance in descending order
                w_stats.sort(key=lambda tup: tup[2], reverse=True)
            else:
                w_stats.sort(key=lambda tup: tup[1], reverse=True)
            w = [np.asarray(tup[0]).reshape(k1, k2) for tup in w_stats[:l]]
        else:
            # using special selection method
            # TODO:
            w = []
            raise NotImplementedError("only support variance and magnitude currently")

    return w


def conv_layer(one_image, w):
        # I_l -> convolve with lth filter
        I = [ndimage.convolve(one_image, one_w, mode='constant', cval=0.0) for one_w in w]
        return I


def image_prepration(data_loader, num_images=5000, patch_size=(5, 5), stride=2):
    # first prepare the images
    raw_images = []
    labels = []
    for i, data in enumerate(data_loader):
        image, label = data
        labels.append(label)
        image = np.squeeze(image.numpy(), axis=(0, 1))
        raw_images.append(image)
        if len(raw_images) >= num_images:
            break
    print('total images:', len(raw_images))
    print('image shape:', raw_images[0].shape)
    # do first layer
    # We'll get a matrix X in shape (k1k2 x num_images x num_patches)
    X = process_images(raw_images, patch_size=patch_size, stride=stride)
    print('X shape:', X.shape)
    return raw_images, labels, X


def PCANet1(raw_images, labels, X, l, num_images=5000, patch_size=(5, 5), stride=2, block_size=(7, 7), block_stride=3):
    # get PCA filters
    w_l1 = get_w(X, l, patch_size=patch_size, method='PCA')
    # convolution
    I_layer1 = dict()
    conved_images = []
    for i, one_image in enumerate(raw_images):
        I_i = conv_layer(one_image, w_l1)
        I_layer1[i] = I_i
        conved_images.extend(I_i)
    
    int_img_dict = dict()
    # output layer
    for idx in I_layer1:
        curr_images = I_layer1[idx]
        integer_image = None
        for one_image in curr_images:
            binarized_image = binarize(one_image)
            if integer_image is None:
                integer_image = binarized_image
            else:
                for i in range(integer_image.shape[0]):
                    for j in range(integer_image.shape[1]):
                        integer_image[i][j] = (integer_image[i][j] << 1) + binarized_image[i][j]
        #print(idx, ':')
        #print(integer_image)
        int_img_dict[idx] = integer_image

    feats = dict()
    for idx in int_img_dict:
        integer_image = int_img_dict[idx]
        bhist = []
        all_vec_bhist = []
        for i in range(0, integer_image.shape[0]-block_size[0], block_stride):
            for j in range(0, integer_image.shape[1]-block_size[1], block_stride):
                curr_block = integer_image[i:i+block_size[0], j:j+block_size[1]]
                block_hist = dict()
                for one_integer in curr_block.reshape((curr_block.size, )):
                    if one_integer not in block_hist:
                        block_hist[one_integer] = 0
                    block_hist[one_integer] += 1
                bhist.append(block_hist)
                # vectorize bhist
                vec_bhist = np.zeros((1 << l, ))
                for hist_bin in range(0, (1 << l)):
                    vec_bhist[hist_bin] = block_hist.get(hist_bin, 0)
                all_vec_bhist.extend(vec_bhist)
        feats[idx] = all_vec_bhist
    return feats, labels


def FourierNet1(raw_images, labels, X, l, num_images=5000, patch_size=(5, 5), stride=2, block_size=(7, 7), block_stride=3, fourier_basis_sel='variance'):
    # get PCA filters
    w_l1 = get_w(X, l, patch_size=patch_size, method='Fourier', fourier_basis_sel=fourier_basis_sel)
    
    # convolution
    I_layer1 = dict()
    conved_images = []
    for i, one_image in enumerate(raw_images):
        I_i = conv_layer(one_image, w_l1)
        I_layer1[i] = I_i
        conved_images.extend(I_i)
    
    int_img_dict = dict()
    # output layer
    for idx in I_layer1:
        curr_images = I_layer1[idx]
        integer_image = None
        for one_image in curr_images:
            binarized_image = binarize(one_image)
            if integer_image is None:
                integer_image = binarized_image
            else:
                for i in range(integer_image.shape[0]):
                    for j in range(integer_image.shape[1]):
                        integer_image[i][j] = (integer_image[i][j] << 1) + binarized_image[i][j]
        #print(idx, ':')
        #print(integer_image)
        int_img_dict[idx] = integer_image

    feats = dict()
    for idx in int_img_dict:
        integer_image = int_img_dict[idx]
        bhist = []
        all_vec_bhist = []
        for i in range(0, integer_image.shape[0]-block_size[0], block_stride):
            for j in range(0, integer_image.shape[1]-block_size[1], block_stride):
                curr_block = integer_image[i:i+block_size[0], j:j+block_size[1]]
                block_hist = dict()
                for one_integer in curr_block.reshape((curr_block.size, )):
                    if one_integer not in block_hist:
                        block_hist[one_integer] = 0
                    block_hist[one_integer] += 1
                bhist.append(block_hist)
                # vectorize bhist
                vec_bhist = np.zeros((1 << l, ))
                for hist_bin in range(0, (1 << l)):
                    vec_bhist[hist_bin] = block_hist.get(hist_bin, 0)
                all_vec_bhist.extend(vec_bhist)
        feats[idx] = all_vec_bhist
    return feats, labels


def RandNet1(raw_images, labels, X, l, num_images=5000, patch_size=(5, 5), stride=2, block_size=(7, 7), block_stride=3):
    # get PCA filters
    w_l1 = get_w(X, l, patch_size=patch_size, method='Random')
    # convolution
    I_layer1 = dict()
    conved_images = []
    for i, one_image in enumerate(raw_images):
        I_i = conv_layer(one_image, w_l1)
        I_layer1[i] = I_i
        conved_images.extend(I_i)
    
    int_img_dict = dict()
    # output layer
    for idx in I_layer1:
        curr_images = I_layer1[idx]
        integer_image = None
        for one_image in curr_images:
            binarized_image = binarize(one_image)
            if integer_image is None:
                integer_image = binarized_image
            else:
                for i in range(integer_image.shape[0]):
                    for j in range(integer_image.shape[1]):
                        integer_image[i][j] = (integer_image[i][j] << 1) + binarized_image[i][j]
        #print(idx, ':')
        #print(integer_image)
        int_img_dict[idx] = integer_image

    feats = dict()
    for idx in int_img_dict:
        integer_image = int_img_dict[idx]
        bhist = []
        all_vec_bhist = []
        for i in range(0, integer_image.shape[0]-block_size[0], block_stride):
            for j in range(0, integer_image.shape[1]-block_size[1], block_stride):
                curr_block = integer_image[i:i+block_size[0], j:j+block_size[1]]
                block_hist = dict()
                for one_integer in curr_block.reshape((curr_block.size, )):
                    if one_integer not in block_hist:
                        block_hist[one_integer] = 0
                    block_hist[one_integer] += 1
                bhist.append(block_hist)
                # vectorize bhist
                vec_bhist = np.zeros((1 << l, ))
                for hist_bin in range(0, (1 << l)):
                    vec_bhist[hist_bin] = block_hist.get(hist_bin, 0)
                all_vec_bhist.extend(vec_bhist)
        feats[idx] = all_vec_bhist
    return feats, labels    

################################################################################
# Two layers
################################################################################
def PCANet(data_loader, l1, l2, num_images=5000, patch_size=(5, 5), stride=1):
    # first prepare the images
    raw_images = []
    label_dict = []
    for i, data in enumerate(data_loader):
        image, label = data
        label_dict[i] = label
        image = np.sequeeze(image.numpy(), axis=(0, 1))
        raw_images.append(image)
        if len(raw_images) >= num_images:
            break
    # do first layer
    # We'll get a matrix X in shape (k1k2 x num_images x num_patches)
    X = process_images(raw_images, patch_size=patch_size, stride=stride)
    # get PCA filters
    w_l1 = get_w(X, l1, patch_size=patch_size)
    # convolution
    I_layer1 = dict()
    conved_images = []
    for i, one_image in enumerate(raw_images):
        I_i = conv_layer(one_image, w_l1)
        I_layer1[i] = I_i
        conved_images.extend(I_i)
    # each convolution results in I_i should be considered as a new image
    # do second layer
    Y = process_images(conved_images, patch_size=patch_size, stride=stride)
    # get PCA filters
    w_l2 = get_w(Y, l2, patch_size=patch_size)
    # convolution
    O_layer2 = dict()
    for idx_layer1 in I_layer1:
        O_layer2[idx_layer1] = dict()
        I_i = I_layer1[idx_layer1]
        for i, one_image in enumerate(I_i):
            O_idx_layer1_i = conv_layer(one_image, w_l2)
            O_layer2[idx_layer1][i] = O_idx_layer1_i
    
    return O_layer2, label_dict


def output_layer(odict, block_size=(7, 7), stride=3):
    int_img_dict = dict()
    for idx_layer1 in odict:
        layer1_dict = odict[idx_layer1]
        int_img_dict[idx_layer1] = []
        # For each of the L1 input images I_il for the second stage, binarize the L2 outputs.
        for idx_layer2 in layer1_dict:
            layer2_images = layer1_dict[idx_layer2]
            integer_image = None
            for one_image in layer2_images:
                binarized_image = binarize(one_image)
                if integer_image is None:
                    integer_image = binarized_image
                else:
                    for i in range(integer_image.shape[0]):
                        for j in range(integer_image.shape[1]):
                            integer_image[i][j] = (integer_image[i][j] << 1) + binarized_image[i][j]
            int_img_dict[idx_layer1].append(integer_image)
    
    # do blockwise histogram
    feats = dict()
    for idx_layer1 in int_img_dict:
        curr_int_images = int_img_dict[idx_layer1]
        feats[idx_layer1] = []
        for one_image in curr_int_images:
            bhist = []
            for i in range(0, one_image.shape[0]-block_size[0], stride):
                for j in range(0, one_image.shape[1]-block_size[1], stride):
                    curr_block = one_image[i:i+block_size[0], j:j+block_size[1]]
                    block_hist = dict()
                    for one_integer in curr_block.reshape((curr_block.size(), )):
                        if one_integer not in block_hist:
                            block_hist[one_integer] = 0
                        block_hist[one_integer] += 1
                    bhist.append(block_hist)
        
        feats[idx_layer1].append(bhist)
    return feats



if __name__ == '__main__':
    batch_size = 1
    num_images = 5000
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = MNIST(root='../datasets/mnist/', transform=img_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    #feats_PCANet1, labels = PCANet1(data_loader=dataloader, l=2, num_images=num_images, block_size=(7, 7), block_stride=3)
    #feats_RandNet1, labels = RandNet1(data_loader=dataloader, l=2, num_images=num_images, block_size=(7, 7), block_stride=3)
    feats, labels = FourierNet1(data_loader=dataloader, l=6, num_images=num_images, block_size=(7, 7), block_stride=3)


    # do logistic regression
    X = []
    for i in range(num_images):
        X.append(feats[i])
    X = np.asarray(X)
    y = labels
    print('feature matrix shape:', X.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=233)
    clf = LogisticRegression(random_state=233, solver='lbfgs', multi_class='multinomial', max_iter=1000)
    clf.fit(X_train, y_train)

    #y_pred = clf.predict(X_test)
    score = clf.score(X_test, y_test)
    print('mean accuracy score:', score)
