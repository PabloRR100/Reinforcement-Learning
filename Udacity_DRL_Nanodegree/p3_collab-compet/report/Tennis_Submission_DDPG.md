

```python
!pip -q install ./python
!pip install beautifultable
print('Installing packages...')
```

    Collecting beautifultable
      Downloading https://files.pythonhosted.org/packages/4a/22/6ca82099e8cdd356555e25c5550f7fc0c91ceccac36402617cc88f590b9b/beautifultable-0.5.3-py2.py3-none-any.whl
    Installing collected packages: beautifultable
    Successfully installed beautifultable-0.5.3
    [33mYou are using pip version 9.0.1, however version 18.1 is available.
    You should consider upgrading via the 'pip install --upgrade pip' command.[0m
    Installing packages...



```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: pabloruizruiz
Collaboration and Competition - Unity Environment - Tennis Game - DDPG Algorithm
"""

import os
import glob
import time
import torch
import pickle
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from beautifultable import BeautifulTable as BT
```


```python
from agent import Agent
from unityagents import UnityEnvironment
from utils import timeit, count_parameters, create_table

cuda = True if torch.cuda.is_available() else False
gpus = True if torch.cuda.device_count() > 1 else False
```


```python
MAXLEN = 100
EPISODES = 10000
PRINT_EVERY = 100
LEARN_PERIOD = 20
NUM_SAMPLES = 10

ENV = '/data/Tennis_Linux_NoVis/Tennis'
```

### Explore the World


```python
def test_world():
    
    # Test World
    # ----------
    
    env = UnityEnvironment(file_name=ENV)
    
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    
    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)
    print('\nNumber of Agents: ', num_agents)
    
    def take_actions(env_info, brain):        
        num_agents = len(env_info.agents)
        action_size = brain.vector_action_space_size
        actions = np.random.randn(num_agents, action_size) 
        actions = np.clip(actions, -1, 1)                  
        env_info = env.step(actions)[brain_name]           
        next_states = env_info.vector_observations
        return env_info, actions, next_states
    
    action_size = brain.vector_action_space_size
    states = env_info.vector_observations
    state_vector_names = ['racket x pos', 'racket y pos', 'racket x velocity', 'racket y velocity',
                          'ball x pos', 'ball y pos', 'ball x velocity', 'ball y velocity']
    
    print('\nA state vector for one of the agent looks like:')
    state = states[0].reshape(3,8)
    table = create_table(state, state_vector_names)
    print(table)
        
    # But only the last row provides new information to each state, so we could simply get those values
    print('\nKeeping only the last row would be: ')
    state0 = states[0].reshape(3,8)[-1]
    table0 = create_table(state0, state_vector_names)
    print(table0)
    
    # If we take a step in the environment
    env_info, _, next_states = take_actions(env_info, brain)

    print('\n\nTaking 1 action, the state vector would look like::')
    state1 = next_states[0].reshape(3,8)
    table1 = create_table(state1, state_vector_names)
    
    print('t = 0')
    print(table)
    print('t = 1')
    print(table1)
    print('Only the last row is providing new information - It have just shifted \n\n')
    print('Although, we can think than given the three provides temporal relationships \n\n')
    
    
    # If we take another step in the environment
    env_info, _, next_states = take_actions(env_info, brain)
    
    print('\n\nTaking another action to be sure')
    state2 = next_states[0].reshape(3,8)
    table2 = create_table(state2, state_vector_names)
    
    print('t = 0')
    print(table)
    print('t = 1')
    print(table1)
    print('t = 2')
    print(table2)
    print('Yes, only the last row is providing new information !')
    
        
    env.close()
    
    
    # Test Agent
    # ----------
    
    state_size, action_size = brain.vector_observation_space_size, brain.vector_action_space_size
    agent = Agent(num_agents=num_agents, state_size=state_size, action_size=action_size)
    
    print('\n\nCapacity of the Actor (# of parameters): ', count_parameters(agent.actor_local))
    print('Capacity of the Critic (# of parameters): ', count_parameters(agent.critic_local))
    return 

```


```python
test_world()
```


```python

```


```python
# Training
# --------

@timeit
def train(env):
    ''' Trains on an environment '''
    
    global MAXLEN
    global EPISODES
    global PRINT_EVERY
    global LEARN_PERIOD
    global NUM_SAMPLES
    
    table = BT()
    table.column_headers = ['Max Iters', 'Epochs', 'Learn Period', 'Num Samples']
    table.append_row([MAXLEN, EPISODES, LEARN_PERIOD, NUM_SAMPLES])

    print('Loading environmnet...\n')
    print(table)
    env = UnityEnvironment(file_name=ENV)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    print("Using brain {}".format(brain_name))

    ## TODO - Question: Why brain.vector_observation_space_size = 8 but
    # env_info.vector_observations[0] = 24 ?
    # Is it like the environment gives you 3 time steps but the agent only observes 8 at a time right?
    
    print('Loading agent...\n')
    num_agents = len(env_info.agents)
#    state_size, action_size = brain.vector_observation_space_size, brain.vector_action_space_size # State Size = 8
    state_size, action_size = len(env_info.vector_observations[0]), brain.vector_action_space_size # State Size = 24
    agent = Agent(num_agents=num_agents, state_size=state_size, action_size=action_size)
    print('Capacity of the Actor (# of parameters): ', count_parameters(agent.actor_local))
    print('Capacity of the Critic (# of parameters): ', count_parameters(agent.critic_local))    
            
    # try:    
        
    avg_score = []
    last_100_mean = 0
    scores_global = []
    scores_concur = deque(maxlen=MAXLEN)
    
    print('Initializing training...\n')
    for e in range(1, EPISODES+1):
        
        # Initialize Episode
        agent.reset()
        t0 = time.time()            
        scores = np.zeros(num_agents)
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        
        
        #for i in range(MAX_ITERS): # Run episode maximum until MAX_ITERS
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
        
        score = np.mean(scores)         # Avg score of the current episode
        scores_concur.append(score)     # Append to our last_100 list
        scores_global.append(score)     # Append to the global list of all episodes
        avg_score.append(np.mean(scores_concur)) # Append to our smoothing avg list
        
        if e % 250 == 0:
            print('\rEpisode {}, Average last 100 scores: {:.2f}, Episode Duration: {:.2f}, \n'\
                  .format(e, avg_score[-1], deltatime))
        
        # If last 100 episodes average score is the best 100 average seen - Save Models & Update
        if avg_score[-1] > last_100_mean:
            
            # Delete previous models
            models = glob.glob('*.pth')
            for m in models:
                os.remove(m)
                    
            # Save new ones
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor_{}.pth'.format(e))
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic_{}.pth'.format(e))
        last_100_mean = avg_score[-1]
        
        if np.mean(scores_concur) > 0.5:
            print('\n\n Goal Achieved - 100 consecutives episodes with mean > 0.5 at Epoch', e)
    
    print('Closing envionment...\n')
    env.close()
    return agent, scores_global, avg_score

    # except:
        # env.close()
        # return None, None, None
```


```python
# Init Training
# -------------

results = dict()
agent, scores, avg_scores = train(ENV)
```

    Loading environmnet...
    
    +-----------+--------+--------------+-------------+
    | Max Iters | Epochs | Learn Period | Num Samples |
    +-----------+--------+--------------+-------------+
    |    100    | 10000  |      20      |     10      |
    +-----------+--------+--------------+-------------+


    INFO:unityagents:
    'Academy' started successfully!
    Unity Academy name: Academy
            Number of Brains: 1
            Number of External Brains : 1
            Lesson number : 0
            Reset Parameters :
    		
    Unity brain name: TennisBrain
            Number of Visual Observations (per agent): 0
            Vector Observation space type: continuous
            Vector Observation space size (per agent): 8
            Number of stacked Vector Observation: 3
            Vector Action space type: continuous
            Vector Action space size (per agent): 2
            Vector Action descriptions: , 


    Using brain TennisBrain
    Loading agent...
    
    Capacity of the Actor (# of parameters):  39554
    Capacity of the Critic (# of parameters):  105729
    Initializing training...
    
    Episode 250, Average last 100 scores: -0.00, Episode Duration: 0.21, 
    
    Episode 500, Average last 100 scores: -0.00, Episode Duration: 0.21, 
    
    Episode 750, Average last 100 scores: -0.00, Episode Duration: 0.20, 
    
    Episode 1000, Average last 100 scores: 0.00, Episode Duration: 0.21, 
    
    Episode 1250, Average last 100 scores: -0.00, Episode Duration: 0.21, 
    
    Episode 1500, Average last 100 scores: 0.00, Episode Duration: 0.20, 
    
    Episode 1750, Average last 100 scores: 0.02, Episode Duration: 0.50, 
    
    Episode 2000, Average last 100 scores: 0.04, Episode Duration: 0.44, 
    
    Episode 2250, Average last 100 scores: 0.04, Episode Duration: 0.45, 
    
    Episode 2500, Average last 100 scores: 0.06, Episode Duration: 0.51, 
    
    Episode 2750, Average last 100 scores: 0.05, Episode Duration: 0.22, 
    
    Episode 3000, Average last 100 scores: 0.04, Episode Duration: 0.45, 
    
    Episode 3250, Average last 100 scores: 0.03, Episode Duration: 0.51, 
    
    Episode 3500, Average last 100 scores: 0.02, Episode Duration: 0.25, 
    
    Episode 3750, Average last 100 scores: 0.04, Episode Duration: 0.44, 
    
    Episode 4000, Average last 100 scores: 0.04, Episode Duration: 0.45, 
    
    Episode 4250, Average last 100 scores: 0.04, Episode Duration: 0.45, 
    
    Episode 4500, Average last 100 scores: 0.03, Episode Duration: 0.76, 
    
    Episode 4750, Average last 100 scores: 0.04, Episode Duration: 0.43, 
    
    Episode 5000, Average last 100 scores: 0.04, Episode Duration: 0.40, 
    
    Episode 5250, Average last 100 scores: 0.03, Episode Duration: 0.50, 
    
    Episode 5500, Average last 100 scores: 0.04, Episode Duration: 0.44, 
    
    Episode 5750, Average last 100 scores: 0.00, Episode Duration: 0.41, 
    
    Episode 6000, Average last 100 scores: -0.00, Episode Duration: 0.21, 
    
    Episode 6250, Average last 100 scores: -0.00, Episode Duration: 0.21, 
    
    Episode 6500, Average last 100 scores: -0.00, Episode Duration: 0.22, 
    
    Episode 6750, Average last 100 scores: -0.00, Episode Duration: 0.20, 
    
    Episode 7000, Average last 100 scores: -0.00, Episode Duration: 0.21, 
    
    Episode 7250, Average last 100 scores: -0.00, Episode Duration: 0.20, 
    
    Episode 7500, Average last 100 scores: -0.00, Episode Duration: 0.22, 
    
    Episode 7750, Average last 100 scores: -0.00, Episode Duration: 0.22, 
    
    Episode 8000, Average last 100 scores: -0.00, Episode Duration: 0.21, 
    
    Episode 8250, Average last 100 scores: -0.00, Episode Duration: 0.20, 
    
    Episode 8500, Average last 100 scores: -0.00, Episode Duration: 0.20, 
    
    Episode 8750, Average last 100 scores: -0.00, Episode Duration: 0.20, 
    
    Episode 9000, Average last 100 scores: -0.00, Episode Duration: 0.20, 
    
    Episode 9250, Average last 100 scores: 0.00, Episode Duration: 0.21, 
    
    Episode 9500, Average last 100 scores: -0.00, Episode Duration: 0.21, 
    
    Episode 9750, Average last 100 scores: -0.00, Episode Duration: 0.20, 
    
    Episode 10000, Average last 100 scores: -0.00, Episode Duration: 0.22, 
    
    Closing envionment...
    
    'train'  3046861.00 ms



```python
# Save results
# ------------

results['scores'] = scores
results['average'] = avg_scores
with open('results.pkl', 'wb') as output:
    pickle.dump(results, output, protocol=pickle.HIGHEST_PROTOCOL)
    
```


```python
# Load results
# ------------
    
with open('results.pkl', 'rb') as input:
    results = pickle.load(input)

scores = results['scores']
avg_scores = results['average']
```


```python
# Plot results
# ------------

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores, color='blue', label='scores')
plt.plot(np.arange(1, len(avg_scores)+1), avg_scores, color='red', label='average window=100')
plt.hlines(y=0.5, xmin=0, xmax=10000)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
```


![png](output_12_0.png)



```python
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
```


```python
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

```


```python
test(ENV)
```
