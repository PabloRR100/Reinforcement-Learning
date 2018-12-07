#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: pabloruizruiz

Continuous Control - Unity Environment - The Reacher

"""

import time
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from agent import Agent
from unityagents import UnityEnvironment
from utils import timeit, count_parameters

EPISODES = 300
PRINT_EVERY = 100


# Load World
# ----------

env = UnityEnvironment(file_name='Reacher.app')
#env = UnityEnvironment(file_name='Reacher20.app')

brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode=True)[brain_name]
num_agents = len(env_info.agents)


# Load Agent
# ----------

state_size, action_size = brain.vector_observation_space_size, brain.vector_action_space_size
agent = Agent(state_size=state_size, action_size=action_size)

print('Capacity of the Actor (# of parameters): ', count_parameters(agent.actor_local))
print('Capacity of the Critic (# of parameters): ', count_parameters(agent.critic_local))

# Training
# --------

#@timeit
def train(env, n_episodes=EPISODES, print_every=PRINT_EVERY):
    
    last_100_mean = []
    scores_global = []
    scores_concur = deque(maxlen=print_every)
    
    for e in range(1, n_episodes+1):
        
        j = 0
        # Initialize Episode
        scores = np.zeros(num_agents)
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations                  # get the current state (for each agent)
        
        agent.reset()
        
        while True:
            
            # Act in the enviromnet
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]          
            
            # Observe result of the action
            next_states = env_info.vector_observations         
            rewards = env_info.rewards                         
            dones = env_info.local_done   
                     
            # Store score result
            scores += env_info.rewards                         
            
            for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
                agent.step(state, action, reward, next_state, done)
            
            if j % print_every == 0:
                print('\rStep {}\tAverage Score: {:.2f}'.format(j, np.mean(scores)), end="")
            
            if np.any(dones):                                  # End of the episode
                break
            
            states = next_states                               # roll over states to next time step
            j += 1
            
        agent.sampleandlearn()
        
        score = np.mean(scores)
        scores_concur.append(score)
        scores_global.append(score)
        print('\rEpisode {}, Mean last 100 scores: {:.2f}, Mean current score: {:.2f}, \n'\
              .format(e, np.mean(scores_concur), score))
        
        ## TODO: change save every 10 for best_models
        if np.mean(scores_concur) > last_100_mean:
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor_{}.pth'.format(e))
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic_{}.pth'.format(e))
            
        last_100_mean = np.mean(scores_concur)
            
    return scores_global


# Load Trained Agent
# ------------------

agent = Agent(state_size=state_size, action_size=action_size)
agent.actor_local.load_state_dict(torch.load('checkpoint_actor.pth'))
agent.critic_local.load_state_dict(torch.load('checkpoint_critic.pth'))



# Init Training
# -------------

scores = train()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()


