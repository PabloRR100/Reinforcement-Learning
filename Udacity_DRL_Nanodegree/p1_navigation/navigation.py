#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Navigation
----------

1st project - Udacity Deep Reinforcement Learning Nanodegree 
"""

import torch
import pickle
import numpy as np
from utils import timeit
from collections import deque

cuda = True if torch.cuda.is_available() else False
gpus = True if torch.cuda.device_count() > 1 else False


# Load World
from unityagents import UnityEnvironment
env = UnityEnvironment(file_name = 'Banana.app')
SOLVED = 13.0

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
scores = dict()     # Results container

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
agent = DQAgent('DQN_Agent', state_size, action_size, seed, h_layers, LR, BS, BFS, GAMMA, TAU, UE)

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
        solved = np.mean(last_scores) >= SOLVED
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
            torch.save(agent.qnetwork_local.state_dict(), '{}_checkpoint.pkl'.format(agent.name))
            break
        
    return episode_scores
    
## We don't see agents while training
#env_info = env.reset(train_mode=True)[brain_name]
#state = env_info.vector_observations[0]
#
#scores[agent.name] = train(EPISODES, TIMESTEPS, eps_0, eps_F, eps_decay)
#
#    
    
# Double DQN
# ----------

double_agent = DoubleDQAgent('Double_DQN_Agent', state_size, action_size, seed, h_layers, LR, BS, BFS, GAMMA, TAU, UE)
double_agent

env_info = env.reset(train_mode=True)[brain_name]
state = env_info.vector_observations[0]

scores[double_agent.name] = train(double_agent, EPISODES, TIMESTEPS, eps_0, eps_F, eps_decay)


# Prioritized Experience Replay
# -----------------------------


# Dueling DQN
# -----------


# Rainbow
# -------



# Training Analysis
# -----------------

import seaborn as sns
import matplotlib.pyplot as plt

sns.set()

plt.figure()
sns.lineplot(range(len(scores[agent.name])), scores[agent.name])
plt.axhline(y=SOLVED, color='red')
plt.show()


# Observe Trained Agent on the Environment
# ----------------------------------------

model_path = './models_backups/DQN_agent_checkpoint.pkl'

# Load the brain
dqn_agent = DQAgent('DQN_Agent', state_size, action_size, seed, h_layers, LR, BS, BFS, GAMMA, TAU, UE)
if cuda: dqn_agent.qnetwork_local.load_state_dict(torch.load(model_path))
else: dqn_agent.qnetwork_local.load_state_dict(torch.load(model_path, map_location='cpu'))

ddqn_agent = DoubleDQAgent('DQN_Agent', state_size, action_size, seed, h_layers, LR, BS, BFS, GAMMA, TAU, UE)
if cuda: ddqn_agent.qnetwork_local.load_state_dict(torch.load(model_path))
else: ddqn_agent.qnetwork_local.load_state_dict(torch.load(model_path, map_location='cpu'))

with open('./results_backups/{}.pkl'.format(dqn_agent.name), 'rb') as input:
    scores = pickle.load(input)
    
scores['dqn'] = pickle.load('./results_backups/{}.pkl'.format(dqn_agent.name), 'rb')
scores['ddqn'] = pickle.load('./results_backups/{}.pkl'.format(ddqn_agent.name), 'rb')
    

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
    

