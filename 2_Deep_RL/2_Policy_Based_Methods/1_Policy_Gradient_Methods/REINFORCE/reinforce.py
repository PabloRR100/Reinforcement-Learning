#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: pabloruizruiz

PyTorch Distributions
=====================

The distributions package contains parameterizable probability distributions 
and sampling functions. This allows the construction of stochastic computation 
graphs and stochastic gradient estimators for optimization. 
This package generally follows the design of the TensorFlow Distributions package.

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


import numpy as np
from collections import deque



class Policy(nn.Module):
    
    def __init__(self, s_size=4, h_size=16, a_size=2):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, a_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)
    
    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action) # Action, Action-Selection Probability
    
    
def reinforce(env, policy, optimizer, n_episodes=1000, max_t=1000, gamma=1.0, print_every=100):
    scores_deque = deque(maxlen=100)
    scores = []
    
    # Init Episode
    for i_episode in range(1, n_episodes+1):
        
        saved_log_probs = []
        rewards = []
        state = env.reset()
        
        # Start episode (trajectory)
        for t in range(max_t):
            
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            if done:
                break 
        
        # Collect Cumulative Rewaed of the trajectories in that episode
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))
        
        discounts = [gamma**i for i in range(len(rewards)+1)]
        R = sum([a*b for a,b in zip(discounts, rewards)])
        
        # Calculate ∂U(theta) = ∑ P(tau, theta) * ∂(logP(tau,theta)*R(tau)) = 
        # = ∑ P(tau, theta) * ∑(log_π(a|s))*R(tau)
        policy_loss = []
        for log_prob in saved_log_probs:
            policy_loss.append(-log_prob * R)
        policy_loss = torch.cat(policy_loss).sum()  # ∑(log_π(a|s))*R(tau)
        
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        
        if i_episode % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        if np.mean(scores_deque)>=195.0:
            print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_deque)))
            break
        
    return scores