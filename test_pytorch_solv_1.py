import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd

print("Using PyTorch Version %s" %torch.__version__)

np.random.seed(0)
torch.manual_seed(0)



"""
data preprocessing
"""
print("read X Y data frame from csv")
data_x_all = pd.read_csv('data/data_x.csv')
data_y_all = pd.read_csv('data/data_y.csv')

# transfer data into matrices (tensor)
print("convert data to numpy matrix")
Full_Data_all = data_x_all.to_numpy()
Full_Label_all = data_y_all.to_numpy()

# Randomly permute a sequence, or return a permuted range.
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
Full_Data = Full_Data_all[:20000]
Full_Label = Full_Label_all[:20000]

# get dimension
n_sample, n_feature = Full_Data.shape
_, n_label = Full_Label.shape

# split data into training set and testing set
# training data from 0 to 80%
Train_Data = Full_Data[:int(n_sample * 0.95)]
Train_Label = Full_Label[:int(n_sample * 0.95)]
# testing data from 80% to 100%
Test_Data = Full_Data_all[int(n_sample * 0.95):]
Test_Label = Full_Label_all[int(n_sample * 0.95):]



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
#
# net = nn.Sequential(
#     nn.Linear(n_feature, 100),
#     nn.ReLU(),
#     nn.Linear(100, 100),
#     nn.ReLU(),
#     nn.Linear(100, n_label),
#     nn.Sigmoid())
# print(net)

loss_func = nn.BCELoss()
learning_rate = 1e-3
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

train_loss = []
train_accuracy = []
epoch = 200

Y_train_t = torch.FloatTensor(Train_Label).reshape(-1, 1)
X_train_t = torch.FloatTensor(Train_Data)
for i in range(epoch):
    y_hat = net(X_train_t)
    loss = loss_func(y_hat, Y_train_t)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    y_hat_class = np.where(y_hat.detach().numpy() < 0.5, 0, 1)
    accuracy = np.sum(Train_Label.reshape(-1, 1) == y_hat_class) / len(Train_Label)
    train_accuracy.append(accuracy)
    train_loss.append(loss.item())
    print("Epoch: {}, Loss: {}, Accuracy: {}".format(i, loss.item(), accuracy))

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
y_hat_test_class = np.where(y_hat_test.detach().numpy()<0.5, 0, 1)
test_accuracy = np.sum(Test_Label.reshape(-1,1) == y_hat_test_class) / len(Test_Data)
print("Test Accuracy with testing data {}".format(test_accuracy))

# Test using all data
X_test_t = torch.FloatTensor(Full_Data_all)
y_hat_test = net(X_test_t)
y_hat_test_class = np.where(y_hat_test.detach().numpy()<0.5, 0, 1)
test_accuracy = np.sum(Full_Label_all.reshape(-1,1) == y_hat_test_class) / len(Full_Label_all)
print("Test Accuracy with full data {}".format(test_accuracy))

# Test using balanced data
X_test_t = torch.FloatTensor(Full_Data_eq)
y_hat_test = net(X_test_t)
y_hat_test_class = np.where(y_hat_test.detach().numpy()<0.5, 0, 1)
test_accuracy = np.sum(Full_Label_eq.reshape(-1,1) == y_hat_test_class) / len(Full_Label_eq)
print("Test Accuracy with balanced data {}".format(test_accuracy))
