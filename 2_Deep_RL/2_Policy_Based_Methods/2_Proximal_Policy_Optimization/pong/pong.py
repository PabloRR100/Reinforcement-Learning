#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Agent that learns to play pong with Policy Gradient Optimizaion
"""

import gym
import numpy as np


# Initialize World
# ----------------

env = gym.make("Pong-v0")
observation = env.reset() # This gets us the image

prev_x = None # used in computing the difference frame
xs,hs,dlogps,drs = [],[],[],[]

reward_sum = 0
episode_number = 0
running_reward = None

render = False


# Set Agent in the world
# ----------------------

from agent import Agent

D = 80 * 80
HIDDEN = 200
ROLLOUTS = 80 
BATCHSIZE = 10 # Update frequency (episodes)

agent = Agent(D, HIDDEN)

# Play Pong!
# ----------

while True:

    if render: env.render()
    
    # Image differencing
    cur_x = agent.prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(D)
    prev_x = cur_x
    
    # Sampling and action based on our current policy
    aprob, h = agent.policy_forward(x)
    action = 2 if np.random.uniform() < aprob else 3 
    
    