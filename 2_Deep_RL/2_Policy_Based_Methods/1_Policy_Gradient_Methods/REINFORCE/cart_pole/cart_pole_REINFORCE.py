#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: pabloruizruiz
"""

import gym
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from pyvirtualdisplay import Display
display = Display(visible=0, size=(1400, 900))
display.start()

is_ipython = 'inline' in plt.get_backend()
if is_ipython:
    from IPython import display

plt.ion()


# Load World
# ----------


env = gym.make('CartPole-v0')
env.seed(0)
print('observation space:', env.observation_space)
print('action space:', env.action_space)


# Load Agent
# ----------
