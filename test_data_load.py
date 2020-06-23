import numpy as np
import pandas as pd
import torch
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

# split data into training set and testing set
# training data from 0 to 80%
Train_Data = Full_Data[:int(n_sample * 0.8)]
Train_Label = Full_Label[:int(n_sample * 0.8)]
# testing data from 80% to 100%
Test_Data = Full_Data[int(n_sample * 0.8):]
Test_Label = Full_Label[int(n_sample * 0.8):]

Test_Data = torch.FloatTensor(Test_Data)