import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from varlinreg import VarLinReg
from varlinregtf import VarLinRegTf

# load data

with open('boston.pkl', 'rb') as f:
    data = pickle.load(f, encoding='latin1')

X=data["X"]
y=data["y"]

N = X.shape[0] # num data points

# Shuffle the data
idx = np.random.permutation(X.shape[0])
# determine the training set size
Ntr = np.int(N*0.9)

# Split the data into train and test
Xtr = X[idx[0:Ntr],:]  # train input
Xts = X[idx[Ntr:],:]   # test input
ytr = y[idx[0:Ntr]]    # train output
yts = y[idx[Ntr:]]     # test output

# Create model instance
model = VarLinRegTf()

# Train and predict

model.train(Xtr, ytr)
ypred = model.predict(Xts)

ypred_mean = ypred[0] # get the predictive mean

rmse = np.sqrt(np.mean((ypred_mean - yts)**2))

print("Root mean squared error is", rmse)
