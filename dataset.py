import numpy as np
import torch
from torchvision import datasets
from torch.utils.data import Dataset
from PIL import Image
import pickle
import pandas as pd

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
    elif name == 'psse':
        return get_local_csv(name)


def get_local_csv(name):
    """
    data preprocessing
    """
    print("read X Y data frame from csv")
    data_x = pd.read_csv('data/data_x.csv')
    data_y = pd.read_csv('data/data_y.csv')

    # transfer data into matrices (tensor)
    print("convert data to numpy matrix")
    Full_Data = data_x.to_numpy()
    Full_Label = data_y.to_numpy()

    # # Randomly permute a sequence, or return a permuted range.
    indecs = np.random.permutation(len(Full_Data))
    Full_Data = Full_Data[indecs]
    Full_Label = Full_Label[indecs]

    # get dimension
    n_sample, n_feature = Full_Data.shape
    _, n_label = Full_Label.shape

    # create balanced data
    idx_zero = np.where(Full_Label == 0)[0]
    idx_one = np.where(Full_Label == 1)[0]
    idx_zero_part = idx_zero[:len(idx_one)]
    idx_eq = np.concatenate((idx_one, idx_zero_part), axis=None)
    Full_Data_eq = Full_Data[idx_eq]
    Full_Label_eq = Full_Label[idx_eq]

    X_tr = Full_Data
    Y_tr = Full_Label
    X_te = Full_Data_eq
    Y_te = Full_Label_eq

    X_tr = torch.FloatTensor(X_tr)
    Y_tr = torch.FloatTensor(Y_tr)
    X_te = torch.FloatTensor(X_te)
    Y_te = torch.FloatTensor(Y_te)

    return X_tr, Y_tr, X_te, Y_te


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
