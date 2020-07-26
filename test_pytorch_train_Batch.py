import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class torch_data_def(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        return x, y, index

    def __len__(self):
        return len(self.X)

print("Using PyTorch Version %s" %torch.__version__)

np.random.seed(0)
torch.manual_seed(0)


"""
data preprocessing
"""
print("read X Y data frame from csv")
# # full dimensional case
# data_x_all = pd.read_csv('data/data_x.csv')
# data_y_all = pd.read_csv('data/data_y.csv')
# two dimensional case
data_x_all = pd.read_csv('data/data_x_2d.csv')
data_y_all = pd.read_csv('data/data_y_2d.csv')

# transfer data into matrices (tensor)
print("convert data to numpy matrix")
Full_Data_all = data_x_all.to_numpy()
Full_Label_all = data_y_all.to_numpy()

# # Randomly permute a sequence, or return a permuted range.
indecs = np.random.permutation(len(Full_Data_all))
Full_Data_all = Full_Data_all[indecs]
Full_Label_all = Full_Label_all[indecs]

# create balanced data
idx_zero = np.where(Full_Label_all == 0)[0]
idx_one = np.where(Full_Label_all == 1)[0]
idx_zero_part = idx_zero[:len(idx_one)]
idx_eq = np.concatenate((idx_one, idx_zero_part), axis=None)
Full_Data_eq = Full_Data_all[idx_eq]
Full_Label_eq = Full_Label_all[idx_eq]

# we use a small portion
Full_Data = Full_Data_all
Full_Label = Full_Label_all # [:100000]

# get dimension
n_sample, n_feature = Full_Data.shape
_, n_label = Full_Label.shape

# split data into training set and testing set
# training data from 0 to 80%
Train_Data = Full_Data[:int(n_sample * 0.8)]
Train_Label = Full_Label[:int(n_sample * 0.8)]
# testing data from 80% to 100%
Test_Data = Full_Data_all[int(n_sample * 0.8):]
Test_Label = Full_Label_all[int(n_sample * 0.8):]

X_train_t = torch.FloatTensor(Train_Data)
Y_train_t = torch.FloatTensor(Train_Label).reshape(-1, 1)

torch_data = torch_data_def(X_train_t, Y_train_t)
loader_tr = DataLoader(torch_data, batch_size=256, shuffle=False)


# Build your network
net = nn.Sequential(
    nn.Linear(n_feature, 100),
    nn.ReLU(),
    nn.Linear(100, 200),
    nn.ReLU(),
    nn.Linear(200, 200),
    nn.ReLU(),
    nn.Linear(200, 50),
    nn.ReLU(),
    nn.Linear(50, n_label),
    nn.Sigmoid())
print(net)


loss_func = nn.BCELoss()
learning_rate = 1e-3
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

train_loss = []
train_accuracy = []
epoch = 100

for i in range(epoch):

    loss_func = torch.nn.BCELoss()  # ZYC

    for batch_idx, (x, y, idxs) in enumerate(loader_tr):
        y_hat = net(x)
        loss = loss_func(y_hat, y)  # ZYC
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # calculate current accuracy
        y_hat_class = y_hat.round()
        comp = y == y_hat_class
        accuracy = comp.sum().numpy() / len(y)
    print("Epoch: {}, Loss: {}, Accuracy: {}".format(i, loss.item(), accuracy))
    # record
    train_accuracy.append(accuracy)
    train_loss.append(loss.item())

fig, ax = plt.subplots(2, 1, figsize=(12, 8))
ax[0].plot(train_loss)
ax[0].set_ylabel('Loss')
ax[0].set_title('Training Loss')

ax[1].plot(train_accuracy)
ax[1].set_ylabel('Classification Accuracy')
ax[1].set_title('Training Accuracy')

plt.tight_layout()
plt.show()


# Test using testing data
X_test_t = torch.FloatTensor(Test_Data)
y_hat_test = net(X_test_t)
y_hat_test_class = np.where(y_hat_test.detach().numpy() < 0.5, 0, 1)
test_accuracy = np.sum(Test_Label.reshape(-1,1) == y_hat_test_class) / len(Test_Data)
print("Test Accuracy with testing data {}".format(test_accuracy))

# Test using all data
X_test_t = torch.FloatTensor(Full_Data_all)
y_hat_test = net(X_test_t)
y_hat_test_class = np.where(y_hat_test.detach().numpy() < 0.5, 0, 1)
test_accuracy = np.sum(Full_Label_all.reshape(-1,1) == y_hat_test_class) / len(Full_Label_all)
print("Test Accuracy with full data {}".format(test_accuracy))

# Test using balanced data
X_test_t = torch.FloatTensor(Full_Data_eq)
y_hat_test = net(X_test_t)
y_hat_test_class = np.where(y_hat_test.detach().numpy() < 0.5, 0, 1)
test_accuracy = np.sum(Full_Label_eq.reshape(-1,1) == y_hat_test_class) / len(Full_Label_eq)
print("Test Accuracy with balanced data {}".format(test_accuracy))


