import sys
sys.path.append('../code/')
from PCANet2 import PCANet2, pipeline, Parameter
from fourier_utils import visualize_fourier_basis
from basic_utils import get_mnist_basic_data, get_cifar_10_data, show_cifar10_image
import numpy as np
import json
from os import path
import argparse
import pickle
import logger

if __name__ == "__main__":
    RESULT_DIR = "PCANet2_results"
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--config_files",
                        type=str,
                        help='comma seperated json file',
                        required=True)
    parser.add_argument("--run_type", type=str,
                        default="f",
                        help="d: run on demo data, t: run an tiny data set, f: run on full data set")
    args = parser.parse_args()
    config_files = args.config_files
    config_files = config_files.split(",")
    run_flag = args.run_type

    for config_file in config_files:
        # load experiments meta file
        experiment = json.load(open(config_file, 'r'))
        dataset = experiment['dataset']
        if run_flag == 'f':
            name = "mnist-"+dataset
        elif run_flag == 'd':
            name = "mnist-"+dataset+"-demo"
        else:
            name = "mnist-"+dataset + "-tiny"
        params = experiment['params']
        output_dir = path.join(RESULT_DIR, name)
        logger.set_logger_dir(output_dir)

        # load dataset
        if 'cifar' in dataset:
            if not 'gray' in name:
                train_imgs, train_labels = get_cifar_10_data("../datasets/cifar-10-batches-py", train=True)
                test_imgs, test_labels = get_cifar_10_data("../datasets/cifar-10-batches-py", train=False)
                logger.info("successfully load cifar10 data")
                # logger.info("train_img.shape: {}, train_labels.shape: {}, test_imgs.shape: {}, test_labels.shape: {}"\
                #     .format(train_imgs[100:120].shape, train_labels[100:120].shape, test_imgs.shape, test_labels.shape))
                if 'demo' in name:
                    train_imgs, train_labels, test_imgs, test_labels = train_imgs[:500], train_labels[:500], train_imgs[500:600], train_labels[500:600]
                else:
                    train_imgs, train_labels, test_imgs, test_labels = train_imgs[:10000], train_labels[:10000], train_imgs[10000:12000], train_labels[10000:12000]

                logger.info("train_img.shape: {}, train_labels.shape: {}, test_imgs.shape: {}, test_labels.shape: {}"\
                    .format(train_imgs.shape, train_labels.shape, test_imgs.shape, test_labels.shape))

                rgb = True
            else:
                train_imgs, train_labels = get_cifar_10_data("../datasets/cifar-10-batches-py", train=True, gray=True)
                test_imgs, test_labels = get_cifar_10_data("../datasets/cifar-10-batches-py", train=False, gray=True)
                logger.info("successfully load cifar10 data")
                # logger.info("train_img.shape: {}, train_labels.shape: {}, test_imgs.shape: {}, test_labels.shape: {}"\
                #     .format(train_imgs[100:120].shape, train_labels[100:120].shape, test_imgs.shape, test_labels.shape))
                #show_cifar10_image(train_imgs[99])
                if 'demo' in name:
                    train_imgs, train_labels, test_imgs, test_labels = train_imgs[:500], train_labels[:500], train_imgs[500:600], train_labels[500:600]
                else:
                    train_imgs, train_labels, test_imgs, test_labels = train_imgs[:10000], train_labels[:10000], train_imgs[10000:12000], train_labels[10000:12000]

                logger.info("train_img.shape: {}, train_labels.shape: {}, test_imgs.shape: {}, test_labels.shape: {}"\
                    .format(train_imgs.shape, train_labels.shape, test_imgs.shape, test_labels.shape))

                rgb = False

        else:
            trainval_imgs, trainval_labels = get_mnist_basic_data("../datasets/mnist-{}".format(dataset), 
                                            train=True, shuffle=False, log=True)

            # bg-img-rand-demo or bg-img-rand-tiny will run on a small dataset first
            if run_flag == 'd':
                logger.warn("demo mode, only 100 train imgs and 20 test imgs")
                train_imgs, train_labels = trainval_imgs[:100], trainval_labels[:100]
                test_imgs, test_labels = trainval_imgs[100:120], trainval_labels[100:120]
            elif run_flag == 't':
                logger.warn("tiny test, 1000 train imgs and 200 test imgs")
                train_imgs, train_labels = trainval_imgs[:1000], trainval_labels[:1000]
                test_imgs, test_labels = trainval_imgs[10000:10200], trainval_labels[10000:10200]
            else:
                train_imgs, train_labels = trainval_imgs[:10000], trainval_labels[:10000]
                test_imgs, test_labels = trainval_imgs[10000:12000], trainval_labels[10000:12000]
            rgb = False

        experiments_result = []
        for idx, param_dict in enumerate(params):
            if 'skip' in param_dict:
                logger.info("SKIPPING:")
                logger.info(param_dict)
                continue
            param = Parameter(**param_dict)
            
            acc, net, clf = pipeline(train_imgs, train_labels, test_imgs, test_labels, param, log=True, rgb=rgb)
            param_dict['acc'] = acc
            with open(path.join(output_dir, 'filter1_filter2.pkl'), 'wb') as f:
                pickle.dump((net._filters1, net._filters2), f)
            visualize_fourier_basis(np.array(net._filters1), 
                                    net._l1, 
                                    *net._patch_size1, 
                                    save_pic=True, 
                                    pic_name=path.join(output_dir, 
                                                       '{}_{}-{}_{}-{}-{}_{}-{}_{}-filter1'.format(idx, net._method1, *net._patch_size1, net._method2, *net._patch_size2, *net._block_size)))
            if 'cifar' in dataset and not 'gray' in name:
                visualize_fourier_basis(np.array(net._filters1)[..., 0], net._l1, *net._patch_size1, 
                        save_pic=True, pic_name=path.join(output_dir, '{}_{}-{}_{}-{}-{}_{}-{}_{}-filter1_0'.format(idx, net._method1, *net._patch_size1, net._method2, *net._patch_size2, *net._block_size)))
                visualize_fourier_basis(np.array(net._filters1)[..., 1], net._l1, *net._patch_size1, 
                        save_pic=True, pic_name=path.join(output_dir, '{}_{}-{}_{}-{}-{}_{}-{}_{}-filter1_1'.format(idx, net._method1, *net._patch_size1, net._method2, *net._patch_size2, *net._block_size)))
                visualize_fourier_basis(np.array(net._filters1)[..., 2], net._l1, *net._patch_size1, 
                        save_pic=True, pic_name=path.join(output_dir, '{}_{}-{}_{}-{}-{}_{}-{}_{}-filter1_2'.format(idx, net._method1, *net._patch_size1, net._method2, *net._patch_size2, *net._block_size)))
            visualize_fourier_basis(np.array(net._filters2), net._l2, *net._patch_size2, 
                    save_pic=True, pic_name=path.join(output_dir, '{}_{}-{}_{}-{}-{}_{}-{}_{}-filter2'.format(idx, net._method1, *net._patch_size1, net._method2, *net._patch_size2, *net._block_size)))

            with open(path.join(output_dir, '{}_{}-{}_{}-{}-{}_{}-{}_{}.pkl'\
                .format(idx, net._method1, *net._patch_size1, net._method2, *net._patch_size2, *net._block_size)), 'wb') as f:
                pickle.dump(clf, f)
        json.dump(experiment, open(path.join(output_dir, 'result.json'), 'w'))