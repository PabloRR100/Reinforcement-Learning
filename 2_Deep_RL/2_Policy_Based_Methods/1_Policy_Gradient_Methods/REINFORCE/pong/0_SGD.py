#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 09:47:02 2018
@author: pabloruizruiz
"""

# DATASETS
# --------

import numpy as np
import pandas as pd
from random import shuffle

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

def to_df(X, y):
    return pd.concat((pd.DataFrame(X, columns=['X1', 'X2']), 
                pd.DataFrame(y, columns=['y'])), axis=1)

def scatterplot(dfs:list, desat:list):
    assert len(dfs) == len(desat), 'List must be same lenght'
    plt.figure()
    for i in range(len(dfs)):
        sns.scatterplot(x='X1', y='X2', hue='y', data=dfs[i], legend=False,
                        palette=sns.color_palette("Set1", n_colors=2, desat=desat[i]))
    plt.show()

X, y = make_moons(n_samples=5000, random_state=42, noise=0.1)
df = to_df(X, y)
df.head()

scatterplot([df], [1])


# Split the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

df_train = to_df(X_train, y_train)
df_test = to_df(X_test, y_test)

scatterplot([df_train, df_test], [1, 0.1])




# NEURAL NETWORK
# --------------

n_features = 2
n_class = 2
n_iter = 10

class Network():
    '''
    Network for binary classification with:
        1 hidden layer, softmax output and cross-entropy loss function.
    '''
    
    def __init__(self, in_f, out_c, lay_size=100, learning_rate=0.1):
        super(self, Network).__init__()
        
        self.W1 = np.random.rand(in_f, lay_size)
        self.W2 = np.random.rand(lay_size, out_c)
        self.lr = learning_rate
        
    def relu(self, x):
        x[x < 0] = 0
        return x
        
    def softmax(self, x):
        return np.exp(x) / sum(np.exp(x))
    
    def forward(self, x):
        a = x @ self.W1 
        h = self.relu(a)
        z = h @ self.W2
        return self.softmax(z)
        
    def backward(self, xs, hs, loss):
        dW2 = hs.T @ loss
        dh = loss @ self.W2.T
        dh[hs <= 0] = 0
        dW1 = xs.T @ dh
        return dW1, dW2

    def update(self, dW1, dW2):
        self.W1 -= self.lr*dW1
        self.W2 -= self.lr*dW1
        
        
class Optimizer():
    ''' SGD minibatch '''
    
    def __init__(self, Network, n_iter):
        super(self, Optimizer).__ini__()
        self.Net = Network
        self.iters = n_iter
        
    def minibatch_SGD(self, X_train, y_train, minibatch_size):
        for iter in range(self.iters):
            print('Iteration {}'.format(iter))
    
            # Randomize data point
            X_train, y_train = shuffle(X_train, y_train)
    
            for i in range(0, X_train.shape[0], minibatch_size):
                # Get pair of (X, y) of the current minibatch/chunk
                X_train_mini = X_train[i:i + minibatch_size]
                y_train_mini = y_train[i:i + minibatch_size]
    
                self.sgd_step(X_train_mini, y_train_mini)
                
        print('Finished training!')
    
        
    def sgd_step(X_train, y_train):
        ''' Acumulate minibatch info into 1 single pass '''
        grad = self.get_minibatch_grad(X_train, y_train)

        # Update every parameters in our networks (W1 and W2) using their gradients
        for layer in grad:
            # Learning rate: 1e-4
            self.Net[layer] += 1e-4 * grad[layer]
    
        return model
        
    def get_minibatch_grad(self, X_train, y_train):
        ''' Iterate over every input and record the gradient '''
        
        xs, hs, grads = [], [], []
    
        for x, cls_idx in zip(X_train, y_train):
            h, y_pred = self.Net.forward(x)
    
            # Create probability distribution of true label
            y_true = np.zeros(n_class)
            y_true[int(cls_idx)] = 1.
    
            # Compute the gradient of output layer
            grad = y_true - y_pred
    
            # Accumulate the informations of minibatch
            xs.append(x)
            hs.append(h)
            grads.append(grad)
    
        # Backprop using the informations we get from the current minibatch
        return self.backward(np.array(xs), np.array(hs), np.array(losses))
    
    
        

    
        


#x = X[0,:]    
#y = y[0]
#W1 = np.random.rand(in_f, lay_size)
#W2 = np.random.rand(lay_size, out_c)
#    
#h = np.dot(x, W1)
#h = relu(h)
#y = np.dot(h, W2)
#    
#x = softmax()


