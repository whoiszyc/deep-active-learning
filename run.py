import pandas as pd
import numpy as np
from datetime import datetime
import time
import logging
import sys
from dataset import get_dataset, get_handler
from model import get_net
from torchvision import transforms
import torch
from query_strategies import RandomSampling, LeastConfidence, MarginSampling, EntropySampling, \
                                LeastConfidenceDropout, MarginSamplingDropout, EntropySamplingDropout, \
                                KMeansSampling, KCenterGreedy, BALDDropout, CoreSet, \
                                AdversarialBIM, AdversarialDeepFool, ActiveLearningByLearning

def main(para_seed=1, method=None):
    # parameters
    SEED = para_seed

    NUM_INIT_LB = 10000
    NUM_QUERY = 10000
    NUM_ROUND = 20
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 64

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
                     {'n_epoch': 20, 'transform': None,
                      'loader_tr_args': {'batch_size': BATCH_SIZE, 'num_workers': 1},
                      'loader_te_args': {'batch_size': 10000, 'num_workers': 1},
                      'optimizer_args': {'lr': LEARNING_RATE, 'momentum': 0.3}}
                }
    args = args_pool[DATA_NAME]

    # set seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.enabled = False

    # load dataset
    X_tr, Y_tr, X_te, Y_te = get_dataset(DATA_NAME)
    # X_tr = X_tr[:300]
    # Y_tr = Y_tr[:300]

    # start experiment
    n_pool = len(Y_tr)
    n_test = len(Y_te)
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
    net = get_net(DATA_NAME)
    handler = get_handler(DATA_NAME)

    if method == "RandomSampling":
        strategy = RandomSampling(X_tr, Y_tr, idxs_lb, net, handler, args)
    elif method == "LeastConfidence":
        strategy = LeastConfidence(X_tr, Y_tr, idxs_lb, net, handler, args)
    elif method == "MarginSampling":
        strategy = MarginSampling(X_tr, Y_tr, idxs_lb, net, handler, args)
    elif method == "EntropySampling":
        strategy = EntropySampling(X_tr, Y_tr, idxs_lb, net, handler, args)
    elif method == "LeastConfidenceDropout":
        strategy = LeastConfidenceDropout(X_tr, Y_tr, idxs_lb, net, handler, args, n_drop=10)
    elif method == "MarginSamplingDropout":
        strategy = MarginSamplingDropout(X_tr, Y_tr, idxs_lb, net, handler, args, n_drop=10)
    elif method == "EntropySamplingDropout":
        strategy = EntropySamplingDropout(X_tr, Y_tr, idxs_lb, net, handler, args, n_drop=10)
    elif method == "KMeansSampling":
        strategy = KMeansSampling(X_tr, Y_tr, idxs_lb, net, handler, args)
    elif method == "KCenterGreedy":
        strategy = KCenterGreedy(X_tr, Y_tr, idxs_lb, net, handler, args)
    elif method == "BALDDropout":
        strategy = BALDDropout(X_tr, Y_tr, idxs_lb, net, handler, args, n_drop=10)
    elif method == "CoreSet":
        strategy = CoreSet(X_tr, Y_tr, idxs_lb, net, handler, args)
    elif method == "AdversarialBIM":
        strategy = AdversarialBIM(X_tr, Y_tr, idxs_lb, net, handler, args, eps=0.05)
    elif method == "AdversarialDeepFool":
       strategy = AdversarialDeepFool(X_tr, Y_tr, idxs_lb, net, handler, args, max_iter=50)
    elif method == None:
        # customized setup
        strategy = MarginSampling(X_tr, Y_tr, idxs_lb, net, handler, args)
        # strategy = ActiveLearningByLearning(X_tr, Y_tr, idxs_lb, net, handler, args, strategy_list=albl_list, delta=0.1)

    # create logging file
    now = datetime.now()
    dt_string = now.strftime("__%Y_%m_%d_%H_%M")
    FILENAME_CSV = type(strategy).__name__ + dt_string + '.csv'
    FILENAME_LOG = type(strategy).__name__ + dt_string + '.log'
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %I:%M:%S %p',
                        filename=FILENAME_LOG, level=logging.DEBUG)
    logging.info('DATA NAME: ' + DATA_NAME)
    logging.info('STRATEGY: ' + type(strategy).__name__)
    logging.info('SEED: {}'.format(SEED))
    logging.info('NUM_INIT_LB: {}'.format(NUM_INIT_LB))
    logging.info('NUM_QUERY: {}'.format(NUM_QUERY))
    logging.info('NUM_ROUND: {}'.format(NUM_ROUND))
    logging.info('LEARNING_RATE: {}'.format(LEARNING_RATE))
    logging.info('BATCH_SIZE: {}'.format(BATCH_SIZE))

    # record accuracy
    acc_list = []

    # record starting time
    start_time = time.time()

    # predict with untrained model
    P = strategy.predict(X_te, Y_te, flag=1)
    acc_init = 1.0 * (Y_te == P).sum().item() / len(Y_te)
    print('Initial testing accuracy {}'.format(acc_init))
    logging.info('Initial testing accuracy {}'.format(acc_init))
    acc_list.append(acc_init)

    # round 0 accuracy
    strategy.train()

    # testing
    P = strategy.predict(X_te, Y_te)
    acc = 1.0 * (Y_te == P).sum().item() / len(Y_te)
    print('Round 0 testing accuracy {}'.format(acc))
    logging.info('Round 0 testing accuracy {}'.format(acc))
    # record acc to list
    acc_list.append(acc)


    for rd in range(1, NUM_ROUND+1):
        print('Round {}'.format(rd))
        logging.info('Round {}'.format(rd))

        # query
        q_idxs = strategy.query(NUM_QUERY)
        idxs_lb[q_idxs] = True

        # update
        strategy.update(idxs_lb)
        strategy.train()

        # testing
        P = strategy.predict(X_te, Y_te)
        acc = 1.0 * (Y_te == P).sum().item() / len(Y_te)
        print('testing accuracy {}'.format(acc))
        logging.info('testing accuracy {}'.format(acc))

        # record acc to list
        acc_list.append(acc)

    logging.info('learning complete using %s seconds' % (time.time() - start_time))
    logging.info('write accuracy records into csv')
    acc_pd = pd.DataFrame(acc_list)
    acc_pd.to_csv(FILENAME_CSV)

    # close log
    logging.shutdown()

if __name__ == '__main__':
    main(para_seed=1, method='MarginSampling')