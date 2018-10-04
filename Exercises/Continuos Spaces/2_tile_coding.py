#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 20:08:03 2018
@author: pabloruizruiz


Discretizing a continuous space that enables better generalization 
compared to a single grid-based approach.

The fundamental idea is to create several overlapping grids or tilings; 
then for any given sample value, you need only check which tiles it lies in.
 
Then encode the original continuous value by a vector of integer indices or bits 
that identifies each activated tile.


Environment -- Acrobot-v1
=========================

State Space (6):
    The state consists of the sin() and cos() of the two rotational joint
    angles and the joint angular velocities :
    [cos(theta1) sin(theta1) cos(theta2) sin(theta2) thetaDot1 thetaDot2].

Action Space (3):
    The action is either applying +1, 0 or -1 torque on the joint between
    the two pendulum links.

Reward:
    -1 each time step
    0.5 when goal reached
    Acrobot-v1 is an unsolved environment, which means it does not have a 
    specified reward threshold at which it's considered solved.*
    
Initial State:
    From -0.6 to -0.4 with NO velocity
    
Terminal conditions:
    - Reach 0.5 position
    - 200 iterations


"""

import sys
import gym
import numpy as np
import matplotlib.pyplot as plt



# ENVIRONMENT
# -----------


# Create an environment
env = gym.make('Acrobot-v1')
env.seed(505);

# Explore state (observation) space
print("State space:", env.observation_space)
print("- low:", env.observation_space.low)
print("- high:", env.observation_space.high)

# Explore action space
print("Action space:", env.action_space)



# TILINGO F THE STATE SPACE WITH MULTIPLE GRIDS WITH OFFSETS
# ----------------------------------------------------------

# 1 - Create the grids

def create_tiling_grid(low, high, bins=(10,10), offssets=(0.,0.), p=False):
    '''
    Discritze a space by defining multiple grids with different offsets
    '''
    grid = [np.linspace(low[i] + offssets[i], high[i], bins[i]+1)[1:-1] for i in range(len(bins))]
    if p:
        print("Uniform grid: [<low>, <high>] / <bins> => <splits>")
        for l, h, b, splits in zip(low, high, bins, grid):
            print("    [{}, {}] / {} => {}".format(l, h, b, splits))
    return grid

low = [-1.0, -5.0]
high = [1.0, 5.0]
create_tiling_grid(low, high, bins=(10, 10), offsets=(-0.1, 0.5))


