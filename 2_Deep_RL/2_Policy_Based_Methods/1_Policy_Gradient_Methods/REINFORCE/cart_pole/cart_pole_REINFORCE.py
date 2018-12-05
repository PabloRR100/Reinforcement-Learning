#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: pabloruizruiz
"""

import gym
import torch
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

from pyvirtualdisplay import Display
display = Display(visible=0, size=(1400, 900))
display.start()

is_ipython = 'inline' in plt.get_backend()
if is_ipython:
    from IPython import display

plt.ion()


# Load World
# ----------

env = gym.make('CartPole-v0')
env.seed(0)
print('Observation space:', env.observation_space)
print('Action space:', env.action_space)


# Load Agent
# ----------

import sys
sys.path.append('..')
from reinforce import Policy
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

policy = Policy().to(device)
optimizer = optim.Adam(policy.parameters(), lr=1e-2)


# Load Algorithm
# --------------

from reinforce import reinforce
scores = reinforce(env, policy, optimizer)


# Plot Training
# -------------

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()


# Watch Agent
# -----------

env = gym.make('CartPole-v0')

state = env.reset()
img = plt.imshow(env.render(mode='rgb_array'))
for t in range(1000):
    action, _ = policy.act(state)
    img.set_data(env.render(mode='rgb_array')) 
    plt.axis('off')
    display.display(plt.gcf())
    display.clear_output(wait=True)
    state, reward, done, _ = env.step(action)
    if done:
        break 

env.close()