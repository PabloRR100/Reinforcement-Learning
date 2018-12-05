#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Play Pong with a REINFORCE Agent

@author: pabloruizruiz
"""

import gym
import time
import torch
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Load World
# ----------

LEFT=5
RIGHT=4
env = gym.make('PongDeterministic-v4')
print("List of available actions: ", env.unwrapped.get_action_meanings())


# Load Agent
# ----------

from agent import Policy
import torch.optim as optim

agent = Policy().to(device)
optimizer = optim.Adam(agent.parameters(), lr=1e-4)


# Load Parallel Environment
# -------------------------

from pong_utils import parallelEnv, preprocess_batch 
envs = parallelEnv('PongDeterministic-v4', n=4, seed=12345)

# 
def collect_trajectories(envs, agent, tmax=200, nrand=5):
    '''
    Collect trajectories of multiple agents of a parallelized environment
    '''
    n=len(envs.ps)

    state_list=[]
    reward_list=[]
    prob_list=[]
    action_list=[]

    envs.reset()
    envs.step([1]*n) # Start all parallel agents
    
    # perform nrand random steps
    for _ in range(nrand):
        fr1, re1, _, _ = envs.step(np.random.choice([RIGHT, LEFT],n))
        fr2, re2, _, _ = envs.step([0]*n)
    
    for t in range(tmax):

        batch_input = preprocess_batch([fr1,fr2])    
        probs = agent(batch_input).squeeze().cpu().detach().numpy()        
        action = np.where(np.random.rand(n) < probs, RIGHT, LEFT)
        probs = np.where(action==RIGHT, probs, 1.0-probs)
                
        fr1, re1, is_done, _ = envs.step(action)
        fr2, re2, is_done, _ = envs.step([0]*n)

        reward = re1 + re2
        
        # Collect the result
        state_list.append(batch_input)
        reward_list.append(reward)
        prob_list.append(probs)
        action_list.append(action)
        
        # Stop if any of the trajectories is done
        if is_done.any():
            break
        
    return prob_list, state_list, action_list, reward_list

prob, state, action, reward = collect_trajectories(envs, agent, tmax=100)


# Define Surrogagte function
# --------------------------

def states_to_prob(agent, states):
    ''' Network Forward Pass ''' 
    states = torch.stack(states)
    input = states.view(-1,*states.shape[-3:])
    return agent(input).view(states.shape[:-3])


def surrogate(agent, old_probs, states, actions, rewards, discount = 0.995, beta=0.01):
    ''' 
    Modifications to Reinforce Gradient Estiamtion
    Returns ∑ log_prob divided by T === -policy_loss
    '''
    
    # Calculate discounted rewards 
    discount = discount ** np.arange(len(rewards))
    rewards = np.asarray(rewards) * discount[:,np.newaxis]
    
    # Keep only R future
    r_future = rewards[::-1].cumsum(axis=0)[::-1]
    
    # Normalize rewards
    mean = np.mean(r_future, axis=1)
    std = np.std(r_future, axis=1) + 1.0e-10
    rewards_normalized = (r_future - mean[:,np.newaxis])/std[:,np.newaxis]
    
    # Convert everything into pytorch tensors and move to gpu if available
    actions = torch.tensor(actions, dtype=torch.int8, device=device)
    old_probs = torch.tensor(old_probs, dtype=torch.float, device=device)
    rewards = torch.tensor(rewards_normalized, dtype=torch.float, device=device)

    # Convert states to action probabilities - Forward Pass on the Policy Network
    # ---------------------------------------------------------------------------
    new_probs = states_to_prob(agent, states)
    new_probs = torch.where(actions == RIGHT, new_probs, 1.0 - new_probs)
    old_probs = torch.tensor(old_probs).to(device)
    
    # Regularization term
    entropy = -(new_probs*torch.log(old_probs+1.e-10) + (1.0-new_probs)*torch.log(1.0-old_probs+1.e-10))
    return torch.mean(beta*entropy)


# Training
# --------

from parallelEnv import parallelEnv

EPISODES = 800
envs = parallelEnv('PongDeterministic-v4', n=8, seed=1234)

DISC = .99
BETA = .01
TMAX = 320

mean_rewards = []
for e in range(EPISODES):

    # Collect trajectories
    old_probs, states, actions, rewards = collect_trajectories(envs, agent, TMAX)
        
    total_rewards = np.sum(rewards, axis=0)
    
    loss = -surrogate(agent, old_probs, states, actions, rewards, beta=BETA)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    del loss
        
    BETA *= 0.995
    
    # Average reward of the parallel environments
    mean_rewards.append(np.mean(total_rewards))
    
    if (e+1) % 50 == 0:
        print("Episode: {0:d}, score: {1:f}".format(e+1,np.mean(total_rewards)))
        print(total_rewards)    


