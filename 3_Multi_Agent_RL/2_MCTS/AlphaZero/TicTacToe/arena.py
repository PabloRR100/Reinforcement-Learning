#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 09:53:42 2018
@author: pabloruizruiz
"""

import random
from copy import copy


# Create the Game
# ---------------

from ConnectN import ConnectN

game_setting = {'size':(3,3), 'N':3}
game = ConnectN(**game_setting)



# Agent with the Policy
# ---------------------

from Agent import Agent
agent = Agent()


# Create Players
# --------------

import MCTS

def Policy_Player_MCTS(game):
    
    mytree = MCTS.Node(copy(game))
    for _ in range(50):
        mytree.explore(agent)
   
    mytreenext, (v, nn_v, p, nn_p) = mytree.next(temperature=0.1)
    return mytreenext.game.last_move

def Random_Player(game):
    return random.choice(game.available_moves())    


# Training the Agent
# ------------------

import torch
import numpy as np
import progressbar as pb    
import torch.optim as optim
optimizer = optim.Adam(agent.parameters(), lr=.01, weight_decay=1e-4)

losses = []
outcomes = []
episodes = 400

widget = ['training loop: ', pb.Percentage(), ' ', 
          pb.Bar(), ' ', pb.ETA() ]
timer = pb.ProgressBar(widgets=widget, maxval=episodes).start()


for e in range(episodes):

    # Start Tree Game
    mytree = MCTS.Node(ConnectN(**game_setting)) 
    
    vterm = []    # Critic Term
    logterm = []  # Policy Term
    
    # Run 50 moves or until the game is finished
    while mytree.outcome is None:
        for _ in range(50):
            mytree.explore(agent)
            
        # Collect results from exploration and make a move
        current_player = mytree.game.player
        mytree, (v, nn_v, p, nn_p) = mytree.next()        
        mytree.detach_mother()
        
        # compute prob* log pi 
        loglist = torch.log(nn_p)*p
        
        # constant term to make sure if policy result = MCTS result, loss = 0
        constant = torch.where(p>0, p*torch.log(p), torch.tensor(0.))
        logterm.append(-torch.sum(loglist-constant))
        
        vterm.append(nn_v*current_player)
        
    # Compute the "policy_loss"
    outcome = mytree.outcome
    outcomes.append(outcome)
    
    loss = torch.sum( (torch.stack(vterm)-outcome)**2 + torch.stack(logterm) )
    optimizer.zero_grad()

    loss.backward()
    losses.append(float(loss))
    optimizer.step()
    
    if (e+1)%50==0:
        print("Game: ",e+1, ", mean loss: {:3.2f}".format(np.mean(losses[-20:])),
              ", recent outcomes: ", outcomes[-10:])
    del loss
    timer.update(e+1)    
    
timer.finish()

# Training Progress

import matplotlib.pyplot as plt
plt.figure()
plt.plot(losses)
plt.show()



# Play agains AlphaZero
# ---------------------

from Play import Play

# As first player
gameplay = Play(ConnectN(**game_setting), 
              player1 = None, 
              player2 = Policy_Player_MCTS)


# As seconds player
gameplay = Play(ConnectN(**game_setting), 
              player1 = Policy_Player_MCTS, 
              player2 = None)

