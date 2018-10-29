#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Agent that learns to play pong with Policy Gradient Optimizaion
"""

import numpy as np

# Define the Policy Network
# -------------------------
'''
The PN will take a flatten representation of the 2D image of the game
at each frame and returns the probability distribution of taking the actions:
    - UP or DOWN
* Actually it takes the difference between 2 frames, to detect motion
'''

class Agent():
    
    def __init__(self, input, hid):
        self.W1 = np.random.rand(input,hid)
        self.W2 = np.random.rand(hid,1)
        
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))
    
    def prepro(self, I):
        ''' Image preprocessing before entering the network '''
        I = I[35:195]     # crop
        I = I[::2,::2,0]  # downsample
        I[I == 144] = 0   # erase background (background type 1)
        I[I == 109] = 0   # erase background (background type 2)
        I[I != 0] = 1     # everything else (paddles, ball) just set to 1
        return I.astype(np.float).ravel()
    
    def policy_forward(self, x:float):
        h = np.dot(self.W1.T, x) 
        h[h<0] = 0 # ReLU 
        x = np.dot(self.W2.T, h) # Log probability of UP
        p = self.sigmoid(x)  # Squeeze logp [0-1]
        return p, h
    
    def policy_backward(self, eph, epdlogp):
        """ backward pass. (eph is array of intermediate hidden states) """
        dW2 = np.dot(eph.T, epdlogp).ravel()
        dh = np.outer(epdlogp, self.W2)
        dh[eph <= 0] = 0 # backpro prelu
        dW1 = np.dot(dh.T, eph)
        return {'W1':dW1, 'W2':dW2}
    
    
    
HIDDEN = 200
ROLLOUTS = 80 
BATCHSIZE = 10 # Update frequency (episodes)

agent = Agent(6400, HIDDEN)
agent.policy_forward(np.random.rand(6400))
        
x = np.random.rand(5)
w1 = np.random.rand(5)
h = np.dot(w1, x) 
        




import torch.nn as nn

class AgentTorch():
    
    def __init__(self, input, hid):
        
        self.dense1 = nn.Linear(input, hid)
        self.dense2 = nn.Linear(hid, 1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    
    def policy_forward(self, x:float):
        x = self.relu(self.dense1(x))
        x = self.dense(x)
        return 1.0 / (1.0 + np.exp(-x)) # sigmoid to squeeze logp [0-1]
    
    
    