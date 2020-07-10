import pandas as pd
import numpy as np
from datetime import datetime
import time
import logging
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.colors
from dataset import get_dataset, get_handler, get_local_csv
from model import get_net
from torchvision import transforms
import torch
from query_strategies import RandomSampling, LeastConfidence, MarginSampling, EntropySampling, \
                                LeastConfidenceDropout, MarginSamplingDropout, EntropySamplingDropout, \
                                KMeansSampling, KCenterGreedy, BALDDropout, CoreSet, \
                                AdversarialBIM, AdversarialDeepFool, ActiveLearningByLearning


def logger_obj(logger_name, level=logging.DEBUG, verbose=0):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    format_string = ("%(asctime)s - %(levelname)s - %(funcName)s (%(lineno)d):  %(message)s")
    datefmt = '%Y-%m-%d %I:%M:%S %p'
    log_format = logging.Formatter(format_string, datefmt)

    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    if verbose == 1:
        # Creating and adding the console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)

    return logger


def main(X_dir_file, Y_dir_file, para_seed=1, method=None, result_dir=None, visual=False):
    # parameters
    SEED = para_seed

    NUM_INIT_LB = 100
    NUM_QUERY = 20
    NUM_ROUND = 10
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 64
    N_EPOCH = 50

    DATA_NAME = 'psse'
    # DATA_NAME = 'MNIST'
    # DATA_NAME = 'FashionMNIST'
    # DATA_NAME = 'SVHN'
    # DATA_NAME = 'CIFAR10'

    args_pool = {'MNIST':
                    {'n_epoch': 10, 'transform': transforms.Compose([transforms.ToTensor()]), # , transforms.Normalize((0.1307,), (0.3081,))
                     'loader_tr_args':{'batch_size': 64, 'num_workers': 1},
                     'loader_te_args':{'batch_size': 1000, 'num_workers': 1},
                     'optimizer_args':{'lr': 0.01, 'momentum': 0.5}},
                'FashionMNIST':
                    {'n_epoch': 10, 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
                     'loader_tr_args':{'batch_size': 64, 'num_workers': 1},
                     'loader_te_args':{'batch_size': 1000, 'num_workers': 1},
                     'optimizer_args':{'lr': 0.01, 'momentum': 0.5}},
                'SVHN':
                    {'n_epoch': 20, 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))]),
                     'loader_tr_args':{'batch_size': 64, 'num_workers': 1},
                     'loader_te_args':{'batch_size': 1000, 'num_workers': 1},
                     'optimizer_args':{'lr': 0.01, 'momentum': 0.5}},
                'CIFAR10':
                    {'n_epoch': 20, 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]),
                     'loader_tr_args':{'batch_size': BATCH_SIZE, 'num_workers': 1},
                     'loader_te_args':{'batch_size': 1000, 'num_workers': 1},
                     'optimizer_args':{'lr': 0.05, 'momentum': 0.3}},
                 'psse':
                     {'n_epoch': N_EPOCH, 'transform': None,
                      'loader_tr_args': {'batch_size': BATCH_SIZE, 'num_workers': 1},
                      'loader_te_args': {'batch_size': 1000, 'num_workers': 1},
                      'optimizer_args': {'lr': LEARNING_RATE, 'momentum': 0.3}}
                }
    args = args_pool[DATA_NAME]

    # set seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.enabled = False


    # create dir for results
    now = datetime.now()
    dt_string = now.strftime("__%Y_%m_%d_%H_%M")
    # check if the dir is given
    if result_dir is not None:
        # if given, check if the saving directory exists
        # if not given, create dir
        if not os.path.isdir(result_dir):
            os.makedirs(result_dir)
        FILENAME_CSV = result_dir + '/' + method + dt_string + '.csv'
        FILENAME_LOG = result_dir + '/' + method + dt_string + '.log'
    elif result_dir is None:
        # if dir is not given, save results at root dir
        FILENAME_CSV = method + dt_string + '.csv'
        FILENAME_LOG = method + dt_string + '.log'

    # in order to create a new log file at each iteration, we need to create an object
    logger = logger_obj(logger_name=FILENAME_LOG, level=logging.DEBUG)
    # logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %I:%M:%S %p',
    #                     filename=FILENAME_LOG, level=logging.DEBUG)
    logger.info('DATA NAME: ' + DATA_NAME)
    logger.info('STRATEGY: ' + method)
    logger.info('SEED: {}'.format(SEED))
    logger.info('NUM_INIT_LB: {}'.format(NUM_INIT_LB))
    logger.info('NUM_QUERY: {}'.format(NUM_QUERY))
    logger.info('NUM_ROUND: {}'.format(NUM_ROUND))
    logger.info('LEARNING_RATE: {}'.format(LEARNING_RATE))
    logger.info('BATCH_SIZE: {}'.format(BATCH_SIZE))
    logger.info('N_EPOCH: {}'.format(N_EPOCH))

    # load dataset
    #TODO: Load data
    # x_tr, y_tr, x_te, y_te = get_dataset(DATA_NAME)
    x_tr, y_tr, x_te, y_te = get_local_csv(X_dir_file, Y_dir_file, logger, train_split=0.6, test_option=2, flag_normal=1)

    # start experiment
    n_pool = len(y_tr)
    n_test = len(y_te)
    print('number of labeled pool: {}'.format(NUM_INIT_LB))
    print('number of unlabeled pool: {}'.format(n_pool - NUM_INIT_LB))
    print('number of testing pool: {}'.format(n_test))

    # generate initial labeled pool
    # index of the pooled data
    idxs_tmp = np.arange(n_pool)
    # indication of labeled data corresponding to the indices
    idxs_lb = np.zeros(n_pool, dtype=bool)
    # shuffle the indices
    np.random.shuffle(idxs_tmp)
    # get initial samples, the number of which equals to NUM_INIT_LB
    idxs_lb[idxs_tmp[:NUM_INIT_LB]] = True

    # load network
    # TODO: Pytorch Dataset and DataLoader
    net = get_net(DATA_NAME)

    # TODO: define machine learning models
    handler = get_handler(DATA_NAME)

    if method == "RandomSampling":
        strategy = RandomSampling(x_tr, y_tr, idxs_lb, net, handler, args)
    elif method == "LeastConfidence":
        strategy = LeastConfidence(x_tr, y_tr, idxs_lb, net, handler, args)
    elif method == "MarginSampling":
        strategy = MarginSampling(x_tr, y_tr, idxs_lb, net, handler, args)
    elif method == "EntropySampling":
        strategy = EntropySampling(x_tr, y_tr, idxs_lb, net, handler, args)
    elif method == "LeastConfidenceDropout":
        strategy = LeastConfidenceDropout(x_tr, y_tr, idxs_lb, net, handler, args, n_drop=10)
    elif method == "MarginSamplingDropout":
        strategy = MarginSamplingDropout(x_tr, y_tr, idxs_lb, net, handler, args, n_drop=10)
    elif method == "EntropySamplingDropout":
        strategy = EntropySamplingDropout(x_tr, y_tr, idxs_lb, net, handler, args, n_drop=10)
    elif method == "KMeansSampling":
        strategy = KMeansSampling(x_tr, y_tr, idxs_lb, net, handler, args)
    elif method == "KCenterGreedy":
        strategy = KCenterGreedy(x_tr, y_tr, idxs_lb, net, handler, args)
    elif method == "BALDDropout":
        strategy = BALDDropout(x_tr, y_tr, idxs_lb, net, handler, args, n_drop=10)
    elif method == "CoreSet":
        strategy = CoreSet(x_tr, y_tr, idxs_lb, net, handler, args)
    elif method == "AdversarialBIM":
        strategy = AdversarialBIM(x_tr, y_tr, idxs_lb, net, handler, args, eps=0.05)
    elif method == "AdversarialDeepFool":
       strategy = AdversarialDeepFool(x_tr, y_tr, idxs_lb, net, handler, args, max_iter=50)
    elif method == None:
        # customized setup
        strategy = MarginSampling(x_tr, y_tr, idxs_lb, net, handler, args)
        # strategy = ActiveLearningByLearning(X_tr, Y_tr, idxs_lb, net, handler, args, strategy_list=albl_list, delta=0.1)


    # record accuracy
    acc_list = []

    # record starting time
    start_time = time.time()

    # predict with untrained model
    P = strategy.predict(x_te, y_te, logger, flag=1)
    acc_init = 1.0 * (y_te == P).sum().item() / len(y_te)
    print('Initial testing accuracy {}'.format(acc_init))
    logger.info('Initial testing accuracy {}'.format(acc_init))
    acc_list.append(acc_init)

    # round 0 accuracy
    strategy.train(logger)

    # testing
    P = strategy.predict(x_te, y_te, logger)
    acc = 1.0 * (y_te == P).sum().item() / len(y_te)
    print('Round 0 testing accuracy {}'.format(acc))
    logger.info('Round 0 testing accuracy {}'.format(acc))
    # record acc to list
    acc_list.append(acc)

    # colormap
    cmap_1 = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red", "blue"])

    # we enable visualization function if it is a 2-D problem
    if visual == True:
        plt.figure(figsize=(9, 6))
        plt.rcParams.update({'font.family': 'Arial'})
        plt.title('Queried samples through active learning', fontsize=16)
        plt.scatter(x_tr[:5000, 0], x_tr[:5000, 1], c=y_tr[:5000], cmap=cmap_1, alpha=0.1)
        plt.pause(1)


    for rd in range(1, NUM_ROUND+1):
        print('Round {}'.format(rd))
        logger.info('Round {}'.format(rd))

        # query
        print("Querying samples")
        q_idxs = strategy.query(NUM_QUERY, logger)
        idxs_lb[q_idxs] = True
        logger.info("Query {} samples".format(len(q_idxs)))

        # plot queried samples if visual is true assuming a 2-D problem
        if visual == True:
            # df = pd.DataFrame(dict(x=x_tr[q_idxs][:, 0], y=x_tr[q_idxs][:, 1], label=y_tr[q_idxs]))
            # groups = df.groupby('label')
            # for name, group in groups:
            #     plt.plot(group.x, group.y,  marker='o', linestyle='', ms=12, label=name)
            # plt.legend()
            # plt.pause(0.2)
            plt.scatter(x_tr[q_idxs][:, 0], x_tr[q_idxs][:, 1], c=y_tr[q_idxs], cmap=cmap_1)
            plt.pause(1)

        # update
        strategy.update(idxs_lb)
        strategy.train(logger)

        # testing
        P = strategy.predict(x_te, y_te, logger)
        acc = 1.0 * (y_te == P).sum().item() / len(y_te)
        print('testing accuracy {}'.format(acc))
        logger.info('testing accuracy {}'.format(acc))

        # record acc to list
        acc_list.append(acc)

    logger.info('learning complete using %s seconds' % (time.time() - start_time))
    logger.info('write accuracy records into csv')
    acc_pd = pd.DataFrame(acc_list)
    acc_pd.to_csv(FILENAME_CSV, index=False)


if __name__ == '__main__':
    # method_list = ["RandomSampling", "LeastConfidence", 'MarginSampling']
    # for method in method_list:
    #     main(para_seed=1, method=method)

    main('data/data_2d_pq_X.csv', 'data/data_2d_pq_Y.csv', para_seed=5, method="MarginSampling", result_dir="result_2d",  visual=True)