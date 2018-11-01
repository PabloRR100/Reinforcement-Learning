#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Agent that learns to play pong with Policy Gradient Optimizaion
"""

import gym
import numpy as np

'''
Notation
========

- HIDDEN: width of the hidden layer of the network
- Episode: until one of the two players wins 21 times
- BATCHSIZE: after how many epochs the policy is updated

- x: observation / input
- h: hidden state (output of hidden layer for that x)
- xs: observations over the entire episode (unknown lenght)
- hs: hidden states over the entire episode (unkown lenght)


'''


# Initialize World
# ----------------

env = gym.make("Pong-v0")
observation = env.reset() # This gets us the image


prev_x = None 
reward_sum = 0
episode_number = 0
running_reward = None


# Set Agent in the world
# ----------------------

from agent import Agent

D = 80 * 80
HIDDEN = 200
ROLLOUTS = 80 
BATCHSIZE = 10 # Update frequency (episodes)

agent = Agent(D, HIDDEN)
xs,hs,dlogps,drs = [],[],[],[] # Memory needed for RMSProp


# Play Pong!
# ----------

render = False
while True:

    ep_steps = 0
    agent.restart_gradients()
    if render: env.render()
    
    # Image differencing
    cur_x = agent.preprocess(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(D)
    prev_x = cur_x
    
    # Sampling and action based on our current policy
    aprob, h = agent.policy_forward(x)
    action = agent.move(aprob)
    print('Action: ', action)
    
    # Keep the values for future RMSProp gradient computation 
    xs.append(x) 
    hs.append(h) 
    fake_label = 1 if action == 2 else 0 
    dlogps.append(fake_label - aprob) 

    # Move
    observation, reward, done, info = env.step(action)
    reward_sum += reward
    
    drs.append(reward) # record reward 
    
    ep_steps += 1
    
    # If the game is finished:
    if done:
        
        episode_number += 1
        
        # Wrap inputs, hidden states, action grandients and rewards
        ep_x = np.vstack(xs)
        ep_h = np.vstack(hs)
        ep_dlogp = np.vstack(dlogps)
        ep_r = np.vstack(drs)
        
        # Apply discount to the rewards and standardize
        ep_r = agent.normalize_rewards(agent.discount_rewards(ep_r))
        
        # Plug-in the rewards to the gradient function
        ep_dlogp *= ep_r
        grad = agent.policy_backward(ep_x, ep_h, ep_dlogp)
        
        # Sum the gradient for use when we hit the batch size
        agent.aggregate_gradients(grad)
        
        # Update rule if Batch size of examples
        if episode_number % BATCHSIZE == 0:
                agent.update_weights(grad)

        
        xs,hs,dlogps,drs = [],[],[],[]  # Reset memory
        observation = env.reset()       # Reset env
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
        reward_sum = 0
        prev_processed_observations = None
        


    
    
    

