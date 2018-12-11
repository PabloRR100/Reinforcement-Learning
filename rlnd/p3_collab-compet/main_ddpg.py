#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: pabloruizruiz
Collaboration and Competition - Unity Environment - Tennis Game
"""

import time
import torch
import pickle
import numpy as np
#from utils import timer
from collections import deque
import matplotlib.pyplot as plt
from beautifultable import BeautifulTable as BT

from agent import Agent
from unityagents import UnityEnvironment
from utils import timeit, count_parameters

cuda = True if torch.cuda.is_available() else False
gpus = True if torch.cuda.device_count() > 1 else False

MAXLEN = 100
EPISODES = 3000
PRINT_EVERY = 100
LEARN_PERIOD = 20
NUM_SAMPLES = 10

ENV = 'Tennis.app'

test = False
if test:
    
    # Test World
    # ----------
    
    env = UnityEnvironment(file_name=ENV)
    
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    
    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)
    print('Number of Agents: ', num_agents)
    
    states = env_info.vector_observations
    state_vector_names = ['racket x pos', 'racket y pos', 'racket x velocity', 'racket y velocity',
                          'ball x pos', 'ball y pos', 'ball x velocity', 'ball y velocity']
    
    print('A state vector for one of the agent looks like:')
    state = states[0].reshape(3,8)
    table = BT()
    table.column_headers = state_vector_names 
    [table.append_row(state[i].tolist()) for i in range(state.shape[0])]
    print(table)
    
    # But only the last row provides new information to each state, so we could simply get those values
    state = states[0].reshape(3,8)[-1]
    table = BT()
    table.column_headers = state_vector_names 
    table.append_row(state.tolist())
    print(table)
    
    env.close()
    
    
    # Test Agent
    # ----------
    
    state_size, action_size = brain.vector_observation_space_size, brain.vector_action_space_size
    agent = Agent(num_agents=num_agents, state_size=state_size, action_size=action_size)
    
    print('Capacity of the Actor (# of parameters): ', count_parameters(agent.actor_local))
    print('Capacity of the Critic (# of parameters): ', count_parameters(agent.critic_local))

        
# Training
# --------

#@timeit
def train(env):
    ''' Trains on an environment '''
    
    global MAXLEN
    global EPISODES
    global PRINT_EVERY
    global LEARN_PERIOD
    global NUM_SAMPLES
    
    print('Loading environmnet...\n')
    env = UnityEnvironment(file_name=ENV)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    

    
    print('Loading agent...\n')
    num_agents = len(env_info.agents)
    state_size, action_size = brain.vector_observation_space_size, brain.vector_action_space_size
    agent = Agent(num_agents=num_agents, state_size=state_size, action_size=action_size)
    print('Capacity of the Actor (# of parameters): ', count_parameters(agent.actor_local))
    print('Capacity of the Critic (# of parameters): ', count_parameters(agent.critic_local))

    
    last_100_mean = []
    scores_global = []
    scores_concur = deque(maxlen=MAXLEN)
    
    try:
        
        print('Initializing training...\n')
        for e in range(1, EPISODES+1):
            
            # Initialize Episode
            avg_score = list()
            scores = np.zeros(num_agents)
            env_info = env.reset(train_mode=True)[brain_name]
            states = env_info.vector_observations                  # get the current state (for each agent)
            
            agent.reset()
            t0 = time.time()
            
            # Run episode maximum until MAX_ITERS
            #for i in range(MAX_ITERS):
            while True:
                
                # Select an action for each Agent
                actions = agent.act(states)
                env_info = env.step(actions)[brain_name]          
                
                # Observe result of the action
                next_states = env_info.vector_observations         
                rewards = env_info.rewards                         
                dones = env_info.local_done   
                         
                agent.step(states, actions, rewards, next_states, dones)
                
                # Store score result and move on
                scores += env_info.rewards                         
                
                # Roll over states to next time step
                states = next_states                               
                
                # Finish episode when one agent is done                
                if np.any(dones):                                  
                    break                                
            
            
            deltatime = time.time() - t0
            
            score = np.mean(scores)
            scores_concur.append(score)
            scores_global.append(score)
            avg_score.append(scores_concur)
            
            print('\rEpisode {}, Average last 100 scores: {:.2f}, Episode Duration: {:.2f}, \n'\
                  .format(e, avg_score, deltatime))
            
            # If last 100 episodes average score is the best 100 average seen - Save Models
            if avg_score > last_100_mean:
                torch.save(agent.actor_local.state_dict(), 'checkpoint_actor_{}.pth'.format(e))
                torch.save(agent.critic_local.state_dict(), 'checkpoint_critic_{}.pth'.format(e))
            
            # Update current 100 mean            
            last_100_mean = avg_score
        
        print('Closing envionment...\n')
        env.close()
        return agent, scores_global, avg_score
    
    # If errors, close environment 
    except:
        env.close()
        print('There were some error wile training')
        return None, None



# Init Training
# -------------

agent, scores, avg_scores = train(ENV)
with open('results.pickle', 'wb') as output:
    pickle.dump(scores, output, protocol=pickle.HIGHEST_PROTOCOL)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores, color='blue', label='scores')
plt.plot(np.arange(1, len(avg_scores)+1), avg_scores, color='red', label='average window=100')
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()



## Load Trained Agent
## ------------------
#
#agent = Agent(state_size=state_size, action_size=action_size)
#agent.actor_local.load_state_dict(torch.load('checkpoint_actor.pth'))
#agent.critic_local.load_state_dict(torch.load('checkpoint_critic.pth'))
#
#with open('results.pickle', 'rb') as input:
#    scores = pickle.load(input)
#    



# Test Trained Agent
# ------------------

def test(ENV):
    
    print('Loading environmnet...\n')
    env = UnityEnvironment(file_name=ENV)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]    
    env_info = env.reset(train_mode=False)[brain_name] # reset the environment
    states = env_info.vector_observations              # get the current state (for each agent)
    
    print('Loading agent...\n')
    num_agents = len(env_info.agents)
    state_size, action_size = brain.vector_observation_space_size, brain.vector_action_space_size
    agent = Agent(num_agents=num_agents, state_size=state_size, action_size=action_size)
    print('Capacity of the Actor (# of parameters): ', count_parameters(agent.actor_local))
    print('Capacity of the Critic (# of parameters): ', count_parameters(agent.critic_local))
            
    scores = np.zeros(num_agents)                      # initialize the score
    dones = False
    
    # Tranfer Learning
    print('Tranfer Learning into Agent...\n')
    agent.actor_local.load_state_dict(torch.load('checkpoint_actor.pth'))
    agent.actor_local.eval()
    
    agent.critic_local.load_state_dict(torch.load('checkpoint_critic.pth'))
    agent.critic_local.eval()
    
    # Play
    print('Playing...\n')
    while not np.any(dones):
       actions = agent.act(states)                        # select an action (for each agent)
       actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1 
       env_info = env.step(actions)[brain_name]           # send all actions to tne environment
       next_states = env_info.vector_observations         # get next state (for each agent)
       rewards = env_info.rewards                         # get reward (for each agent)
       dones = env_info.local_done                        # see if episode finished
       scores += rewards                                  # update the score (for each agent)
       states = next_states                               # roll over states to next time step
    
    print("Score: {}".format(scores))
    

test(ENV)
