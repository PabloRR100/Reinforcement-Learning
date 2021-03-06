#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Agent that learns to play pong with Policy Gradient Optimizaion
"""

# Define the Policy Network
# -------------------------
'''
The PN will take a flatten representation of the 2D image of the game
at each frame and returns the probability distribution of taking the actions:
    - UP or DOWN
* Actually it takes the difference between 2 frames, to detect motion


Matrices dimensions:
====================

Assuming HID = 200

x  = (1,6400)   -> 1 observation of 6400 
W1 = (6400,HID) -> Matrix to proyect into new space
h  = (1, 200)   -> 1 proyected observation into a space of 200 columns
W2 = (200, 1)   -> Matrix to proyect into 1 unique value (prob of UP)

'''


## TODO: Implement different optimizers from scratch to fully understand them 


# Numpy Implementation
import numpy as np

class Agent():
    
    def __init__(self, input, hid, gam=0.99, eps=1e-5, lr=1e-4, dw=0.99):
        # Weights
        self.W1 = np.random.rand(input,hid)
        self.W2 = np.random.rand(hid,1)      
        self.layers = {'W1': self.W1, 
                       'W2': self.W2}
        
        self.a = None ## backup variable to check gradients
        self.b = None ## are being properly updated
        self.g = None ## 
        
        # Training parameters
        self.gamma = gam        
        self.epsilon = eps
        self.learning_rate = lr
        self.decay_weight = dw
        
        # Gradient memory
        self.exp_g2 = {}
        self.g_dict = {}
        
    def preprocess(self, I):
        ''' Image preprocessing before entering the network '''
        def crop(x):
            return x[35:195]
        
        def downsample(x):
            return x[::2,::2,0]
        
        def erase_background(x):
            x[x == 144] = 0
            x[x == 109] = 0
            x[x != 0] = 1
            return x
        
        I = erase_background(downsample(crop(I)))
        return I.astype(np.float).ravel()
    
    def move(self, p):
        random_value = np.random.uniform()
        if random_value < p: return 2 # UP
        else: return 3 # DOWN
        
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))
    
    def policy_forward(self, x:float):
        ''' Returns the probability of taken an action and the hidden state '''
        h = np.dot(x, self.W1) 
        h[h<0] = 0 # ReLU 
        x = np.dot(h, self.W2) # Log probability of UP
        p = self.sigmoid(x)  # Squeeze logp [0-1]
        return p, h
    
    def policy_backward(self, ep_x, ep_h, ep_dlogp):
        ''' Backward pass. Calculate the gradients '''
        a = np.dot(ep_h.T, ep_dlogp) ##
        dW2 = np.dot(ep_h.T, ep_dlogp).ravel()
        dh = np.outer(ep_dlogp, self.W2)
        dh[ep_h <= 0] = 0 
        dW1 = np.dot(dh, ep_x)
        return {'W1':dW1, 'W2':dW2}
        
    def restart_gradients(self):
        for layer_name in self.layers.keys():
            self.exp_g2[layer_name] = np.zeros_like(self.layers[layer_name])
            self.g_dict[layer_name] = np.zeros_like(self.layers[layer_name])
            
    def aggregate_gradients(self, grad):
        for layer_name in grad:
            self.g_dict[layer_name] += grad[layer_name]        

    def discount_rewards(self, r):
        ''' Return a discounted reward array from the original '''
        running_add = 0
        discounted_r = np.zeros_like(r)
        for t in reversed(range(0, r.size)):
          if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
          running_add = running_add * self.gamma + r[t]
          discounted_r[t] = running_add
        return discounted_r
    
    def normalize_rewards(self, r):
        ''' Normalize a reward array to force half choices being good/bad '''
        r -= np.mean(r)
        r /= np.std(r)
        return r
    
    def update_weights(self):
        for layer_name in self.layers.keys():
            g = self.g_dict[layer_name]
            self.exp_g2[layer_name] = self.decay_rate * self.exp_g2[layer_name] + (1 - self.decay_rate) * g**2
            self.layers[layer_name] += (self.learning_rate * g)/(np.sqrt(self.exp_g2[layer_name] + self.epsilon))
            self.g_dict[layer_name] = np.zeros_like(self.layers[layer_name]) # reset batch gradient memory
