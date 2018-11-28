#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Navigation
----------

1st project - Udacity Deep Reinforcement Learning Nanodegree 
"""

import torch
import numpy as np
from unityagents import UnityEnvironment

# Load World
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

# Hyperparameters 
# ---------------

LR = 5e-4               # learning rate 
GAMMA = 0.99            # discount factor
BATCH_SIZE = 64         # minibatch size

TAU = 1e-3              # for soft update of target parameters
UPDATE_EVERY = 4        # how often to update the network
BUFFER_SIZE = int(1e5)  # replay buffer size



# Create Agent
# ------------

from dqn import DQAgent
agent = DQAgent()

