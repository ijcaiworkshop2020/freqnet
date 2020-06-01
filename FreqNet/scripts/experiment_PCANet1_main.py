import sys
sys.path.append('../code/')

from PCANet import PCANet, Parameter, pipeline
from fourier_utils import visualize_fourier_basis
from basic_utils import get_mnist_basic_data

import numpy as np

import json
import os
from os import path
import argparse
import pickle

import logger

if __name__ == "__main__":
    RESULT_DIR = "PCANet1_results"
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--config_files", type=str, help='comma seperated json file', required=True)
    parser.add_argument("--run_type", type=str, default='f', help='d: run on demo data, t: run on tiny dataset, f: run on full dataset')
    args = parser.parse_args()
    config_files = args.config_files
    config_files = config_files.split(",")
    run_flag = args.run_type

    for config_file in config_files:
        
        # load experiments meta file
        experiment = json.load(open(config_file, 'r'))
        dataset = experiment['dataset']
        params = experiment['params']

        if run_flag == 'f':
            name = 'mnist-' + dataset
        elif run_flag == 'd':
            name = 'mnist-' + dataset + '-demo'
        else:
            name = 'mnist-' + dataset + '-tiny'

        output_dir = path.join(RESULT_DIR, name)
        logger.set_logger_dir(output_dir)

        # load dataset
        trainval_imgs, trainval_labels = get_mnist_basic_data("../datasets/mnist-{}".format(dataset), 
                                train=True, shuffle=False, log=True)
        
        if run_flag == 'd':
            logger.warn("demo mode, only 100 train imgs and 20 test imgs")
            train_imgs, train_labels = trainval_imgs[:100], trainval_labels[:100]
            test_imgs, test_labels = trainval_imgs[100:120], trainval_labels[100:120]
        elif run_flag == 't':
            train_imgs, train_labels = trainval_imgs[:1000], trainval_labels[:1000]
            test_imgs, test_labels = trainval_imgs[10000:10200], trainval_labels[10000:10200]
        else:
            train_imgs, train_labels = trainval_imgs[:10000], trainval_labels[:10000]
            test_imgs, test_labels = trainval_imgs[10000:12000], trainval_labels[10000:12000]

        experiments_result = []
        for idx, param_dict in enumerate(params):
            param = Parameter(**param_dict)
            acc, net, clf = pipeline(train_imgs, train_labels, test_imgs, test_labels, param, log=True)
            param_dict['acc'] = acc

            with open(path.join(output_dir, '{}_{}-{}_{}-{}_{}.pkl'\
                .format(idx, net._method, net._patch_size[0], net._patch_size[1], net._block_size[0], net._block_size[1])), 'wb') as f:
                pickle.dump(clf, f)
            if net._filters:
                visualize_fourier_basis(np.array(net._filters), net._l, *net._patch_size, 
                        save_pic=True, pic_name=path.join(output_dir, '{}-filter'.format(net._l)))
            Z
        json.dump(experiment, open(path.join(output_dir, 'result.json'), 'w'))
