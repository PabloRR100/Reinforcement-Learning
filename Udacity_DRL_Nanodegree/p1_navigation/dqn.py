#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep Q-Learning Algorithm

From Udacity Deep Reinforcement Learning Nanodegree Materials
https://eu.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893
"""

import random
import numpy as np
from utils import count_parameters

from collections import deque
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


cuda = True if torch.cuda.is_available() else False
gpus = True if torch.cuda.device_count() > 1 else False


# Brains
# ------

class DQNet(nn.Module):
    ''' Actor Model Deep Q-Learning '''
    def __init__(self, state_size, action_size, seed, h_layers):
        super(DQNet, self).__init__()     
        self.seed = torch.manual_seed(seed)
        self.n_lay = len(h_layers)
        
        self.fcI = nn.Linear(state_size, h_layers[0])
        self.fcH = nn.ModuleList([nn.Linear(h_layers[l], h_layers[l+1]) for l in range(self.n_lay - 1)])        
        self.fcO = nn.Linear(h_layers[-1], action_size)
        
    def forward(self, state):
        x = F.relu(self.fcI(state))           
        for l in range(self.n_lay - 1):
            x = F.relu(self.fcH[l](x))
        x = F.relu(self.fcO(x))
        return x
  
    
class DuelingQNEt(nn.Module):
    ''' Dueling Q-Network '''
    def __init__(self, state_size, action_size, seed, h_layers):
        super(DQNet, self).__init__()     
        self.seed = torch.manual_seed(seed)
        self.n_lay = len(h_layers)
        
        self.fcI = nn.Linear(state_size, h_layers[0])
        self.fcH = nn.ModuleList([nn.Linear(h_layers[l], h_layers[l+1]) for l in range(self.n_lay - 1)])        
        self.fcO = nn.Linear(h_layers[-1], action_size)
        
    def forward(self, state):
        value = None
        advantage = None
        q = value + advantage
        return q


# Memory
# ------ 

class ReplayBuffer:
    ''' Memory Buffer to sample past experiences '''
    def __init__(self, action_size, buffer_size, batch_size, seed, device):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = device
     
    def add(self, state, action, reward, next_state, done):
        ''' Add a new experience tuple (S,A,R,S',D) '''
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        ''' Randomly sample a batch of experiences '''
        experiences = random.sample(self.memory, k=self.batch_size)

        ## TODO: refactor these tensors
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        ''' Current memory size '''
        return len(self.memory)

# Bodies 
# ------

class DQAgent():
    ''' Interacts with and learns from the environment '''
    
    def __init__(self, name, state_size, action_size, seed, h_layers, LR, BS, BFS, gamma, tau, ue):
        
        self.name = name
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        self.lr = LR
        self.batch_size = BS
        self.buffer_size = BFS
        self.gamma = gamma
        self.tau = tau
        self.update_every = ue
        
        self.criterion = nn.MSELoss().cuda() if cuda else nn.MSELoss()
        self.device = torch.device('cuda' if cuda else 'cpu')


        # Brain: 2 DQNetworks - stimated and fixed targets
        self.qnetwork_local = DQNet(state_size, action_size, seed, h_layers).to(self.device)
        self.qnetwork_target = DQNet(state_size, action_size, seed, h_layers).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)

        # Memory: Replay buffer
        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, self.seed, self.device)
        
        # Initialize time step
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    def act(self, state, eps=0.):
        ''' Returns actions for given state as per current policy '''
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        ''' Update parameters using batch experience (s, a, r, s', done) tuples '''
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # loss = F.mse_loss(Q_expected, Q_targets)
        loss = self.criterion(Q_expected, Q_targets)        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
    def __repr__(self):
        return 'DQN - Agent \n Brain Power: {} Neurons \n Brain Structure: \n {}' \
            .format(count_parameters(self.qnetwork_local), self.qnetwork_local)
        

class DoubleDQAgent(DQAgent):
    ''' Double Deep Q-Learning Actor '''
    
    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        
        # Calculate Target
        self.qnetwork_target.eval()     # Set to inference mode
        with torch.no_grad():
            Q_dash_local = self.qnetwork_local(next_states)
            Q_dash_target = self.qnetwork_target(next_states)
            
            argmax_action  = torch.max(Q_dash_local, dim=1, keepdim=True)[1]
            Q_dash_max = Q_dash_target.gather(1, argmax_action)
            
            y = rewards + gamma * Q_dash_max * (1 - dones)
        self.qnetwork_target.train()    # Put back in train mode
        
        # Predict Q-Value
        self.optimizer.zero_grad()
        Q = self.qnetwork_local(states)
        y_pred = Q.gather(1, actions)            
        
        # Calculate TD-Error
        loss = self.criterion(y_pred, y)
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)
        
    def __repr__(self):
       return 'Double DQN - Agent \n Brain Power: {} Neurons \n Brain Structure: \n {}' \
           .format(count_parameters(self.qnetwork_local), self.qnetwork_local)
        
        
        
            
    
    
    