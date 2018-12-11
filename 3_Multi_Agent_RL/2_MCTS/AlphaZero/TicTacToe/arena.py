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

from agent import Agent
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


# Training the Agent - Alpha Zero Algorithm
# -----------------------------------------



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

