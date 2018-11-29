r#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Navigation
----------

1st project - Udacity Deep Reinforcement Learning Nanodegree 
"""

import torch
import numpy as np
from utils import timeit
from collections import deque

# Load World
from unityagents import UnityEnvironment
env = UnityEnvironment(file_name = 'Banana.app')

# Load Brain (default)
brain_name = env.brain_names[0] # 'BananaBrain'
brain = env.brains[brain_name]  # <unityagetns.brain.BrainParameters'

'''
INFO:unityagents:
'Academy' started successfully!
Unity Academy name: Academy
        Number of Brains: 1
        Number of External Brains : 1
        Lesson number : 0
        Reset Parameters :
                
Unity brain name: BananaBrain
        Number of Visual Observations (per agent): 0
        Vector Observation space type: continuous
        Vector Observation space size (per agent): 37
        Number of stacked Vector Observation: 1
        Vector Action space type: discrete
        Vector Action space size (per agent): 4
        Vector Action descriptions: , , , 
'''


# World Parameters
# ----------------
action_size = brain.vector_action_space_size
state_size = len(env.reset(train_mode=True)[brain_name].vector_observations[0])


# Hyperparameters 
# ---------------

seed = 2018

h_layers=[32,32]    # Hidden Layers
LR = 5e-4           # Learning rate 
BS = 64             # Minibatch size
GAMMA = 0.99        # Discount factor
    
UE = 4              # How often to update the network
TAU = 1e-3          # For soft update of target parameters
BFS = int(1e5)      # Memory capacity


EPISODES = 2000     # Training episodes
TIMESTEPS = 1000    # Max # of timesteps per episode
eps_0 = 1.0         # Initial exploration (epsilon)
eps_F = 0.01        # Minimal exploration
eps_decay = 0.995   # Exploration decay rate


# Create Agent
# ------------

from dqn import DQAgent, DoubleDQAgent
agent = DQAgent(state_size, action_size, seed, h_layers, LR, BS, BFS, GAMMA, TAU, UE)
agent = DoubleDQAgent(state_size, action_size, seed, h_layers, LR, BS, BFS, GAMMA, TAU, UE)

agent


# Training 
# --------

@timeit
def train(agent, EPISODES, TIMESTEPS, eps_0, eps_F, eps_decay):
    eps = eps_0
    episode_scores = list()
    last_scores = deque(maxlen=100)
    def monitor(e,s):
        if e % 100 == 0: 
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(e, np.mean(s)), end="")
        
    def solved(e,s):
        solved = np.mean(last_scores) >= 13.0
        if solved:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}' \
                  .format(e-100, np.mean(s)))
            return True
        return False
    
    # Run episodes
    for e in range(EPISODES):
        
        # Reset the environment
        score = 0
        env_info = env.reset(train_mode=True)[brain_name]
        state = env.reset(train_mode=True)[brain_name].vector_observations[0]
        
        # Run timesteps within the episodes
        for t in range(TIMESTEPS):
            
            # Take an action, reach new state and observe respective reward
            action = agent.act(state, eps).astype(int)
            env_info = env.step(action)[brain_name]       
            next_state = env_info.vector_observations[0]    # What's next state? 
            reward = env_info.rewards[0]                    # What's the reward?
            done = env_info.local_done[0]                   # Is it finished?
            
            # Advance
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done: break
        
        # Episode stats
        last_scores.append(score)
        episode_scores.append(score)
        
        # Decrease exploration (epsilon)
        eps = max(eps_F, eps_decay*eps)
        
        # Monitoring
        monitor(e, last_scores)
        if solved(e,score):
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pkl')
            break
        
    return episode_scores
    
# We don't see agents while training
env_info = env.reset(train_mode=False)[brain_name]
state = env_info.vector_observations[0]

scores = train(EPISODES, TIMESTEPS, eps_0, eps_F, eps_decay)


# Training Analysis
# -----------------

import seaborn as sns
import matplotlib.pyplot as plt

#sns.set_style('darkgrid')
sns.set()
fake = np.random.randn(1000).tolist()

plt.figure()
sns.lineplot(range(len(fake)), fake)
plt.axhline(y=0.95, color='red')
plt.show()


# Observe Trained Agent on the Environment
# ----------------------------------------

# Load the brain
agent = DQAgent
agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

for i in range(5):
    
    env_info = env.reset(train_mode=False)[brain_name]
    state = env_info.vector_observations[0]
    
    for j in range(200):
        action = agent.act(state).astype(int)
        env_info = env.step(action)[brain_name]
        state = env_info.vector_observations[0]  
        done = env_info.local_done[0]
        if done:
            break
    
    
    
        
        
    

    
 


