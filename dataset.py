import numpy as np
import torch
from torchvision import datasets
from torch.utils.data import Dataset
from PIL import Image
import pickle
import pandas as pd
import logging

def get_dataset(name):
    if name == 'MNIST':
        return get_MNIST()
    elif name == 'FashionMNIST':
        return get_FashionMNIST()
    elif name == 'SVHN':
        return get_SVHN()
    elif name == 'CIFAR10':
        return get_CIFAR10()
    elif name == 'iris':
        return get_local_pkl(name)


def normalize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_norm = (X.T - mean[:, None]).T
    X_norm = (X_norm.T / std[:, None]).T
    X_norm[np.isnan(X_norm)] = 0
    return X_norm, mean, std


def get_local_csv(X_dir_file, Y_dir_file, logger, train_split=0.8, test_option=0, flag_normal=0):
    """
    data preprocessing
    """
    print("read X Y data frame from csv")
    x = pd.read_csv(X_dir_file)
    y = pd.read_csv(Y_dir_file)

    # transfer data into matrices (tensor)
    print("convert data to numpy matrix")
    x = x.to_numpy()
    y = y.to_numpy()

    # randomly permute a sequence, or return a permuted range.
    indecs = np.random.permutation(len(x))
    x = x[indecs]
    y = y[indecs]

    # data normalization is very important
    if flag_normal == 1:
        logger.info("Normalize dataset x")
        x, x_mean, x_std = normalize(x)

    # get dimension
    n_sample, n_feature = x.shape
    _, n_label = y.shape

    # add logger
    logger.info("Sample number: ".format(n_sample))
    logger.info("Feature number: ".format(n_feature))
    logger.info("Label number: ".format(n_label))

    # create balanced data
    print("create balanced data")
    logger.info("create balanced data")
    idx_zero = np.where(y == 0)[0]
    idx_one = np.where(y == 1)[0]
    idx_zero_part = idx_zero[:len(idx_one)]
    idx_eq = np.concatenate((idx_one, idx_zero_part), axis=None)
    x_bal = x[idx_eq]
    y_bal = y[idx_eq]
    logger.info("Balance data numbers: ".format(len(y_bal)))

    # split training and testing data
    # training set
    print("Using {} for training".format(n_sample * train_split))
    logger.info("Using {} for training".format(n_sample * train_split))
    x_tr = x[:int(n_sample * train_split)]
    y_tr = y[:int(n_sample * train_split)]

    if test_option == 0:
        logger.info("Test option 0: using {}% portion from the entire dataset".format((1-train_split)*100))
        X_te = x[int(n_sample * (1 - train_split)):]
        y_te = y[int(n_sample * (1 - train_split)):]
    elif test_option == 1:
        logger.info("Test option 1: using all balance dataset")
        X_te = x_bal
        y_te = y_bal
    elif test_option == 2:
        logger.info("Test option 2: using the entire dataset")
        X_te = x
        y_te = y
    else:
        print("Must specify a testing option")

    x_tr = torch.FloatTensor(x_tr)
    y_tr = torch.FloatTensor(y_tr)
    X_te = torch.FloatTensor(X_te)
    y_te = torch.FloatTensor(y_te)

    # we change label data to long tensor if we consider it as a multi-label classification
    y_tr = y_tr.transpose(0, 1)
    y_tr = y_tr.long()
    y_te = y_te.transpose(0, 1)
    y_te = y_te.long()
    y_tr = y_tr[0]
    y_te = y_te[0]

    return x_tr, y_tr, X_te, y_te


def get_local_pkl(name):
    # open a file, where you stored the pickled data
    file = open(name, 'rb')
    # dump information to that file
    data = pickle.load(file)
    # close the file
    file.close()
    print('Loading the pickled data complete')
    data_size = len(data['data'])
    data_size_tr = data_size * 0.8
    X_tr = data['data'][0:data_size_tr]
    Y_tr = data['target'][0:data_size_tr]
    X_te = data['data']
    Y_te = data['target']
    return X_tr, Y_tr, X_te, Y_te

def get_MNIST():
    raw_tr = datasets.MNIST('./MNIST', train=True, download=True)
    raw_te = datasets.MNIST('./MNIST', train=False, download=True)
    X_tr = raw_tr.train_data
    Y_tr = raw_tr.train_labels
    X_te = raw_te.test_data
    Y_te = raw_te.test_labels
    return X_tr, Y_tr, X_te, Y_te

def get_FashionMNIST():
    raw_tr = datasets.FashionMNIST('./FashionMNIST', train=True, download=True)
    raw_te = datasets.FashionMNIST('./FashionMNIST', train=False, download=True)
    X_tr = raw_tr.train_data
    Y_tr = raw_tr.train_labels
    X_te = raw_te.test_data
    Y_te = raw_te.test_labels
    return X_tr, Y_tr, X_te, Y_te

def get_SVHN():
    data_tr = datasets.SVHN('./SVHN', split='train', download=True)
    data_te = datasets.SVHN('./SVHN', split='test', download=True)
    X_tr = data_tr.data
    Y_tr = torch.from_numpy(data_tr.labels)
    X_te = data_te.data
    Y_te = torch.from_numpy(data_te.labels)
    return X_tr, Y_tr, X_te, Y_te

def get_CIFAR10():
    data_tr = datasets.CIFAR10('./CIFAR10', train=True, download=True)
    data_te = datasets.CIFAR10('./CIFAR10', train=False, download=True)
    X_tr = data_tr.train_data
    Y_tr = torch.from_numpy(np.array(data_tr.train_labels))
    X_te = data_te.test_data
    Y_te = torch.from_numpy(np.array(data_te.test_labels))
    return X_tr, Y_tr, X_te, Y_te

def get_handler(name):
    if name == 'MNIST':
        return DataHandler1
    elif name == 'FashionMNIST':
        return DataHandler1
    elif name == 'SVHN':
        return DataHandler2
    elif name == 'CIFAR10':
        return DataHandler3
    elif name == 'psse':
        return DataHandler4

class DataHandler1(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(x.numpy(), mode='L')
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)

class DataHandler2(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(np.transpose(x, (1, 2, 0)))
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)

class DataHandler3(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(x)
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)

class DataHandler4(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)
