#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 20:08:47 2018
@author: pabloruizruiz
"""

import sys
import gym
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
np.set_printoptions(precision=3, linewidth=120)

from pyvirtualdisplay import Display
display = Display(visible=0, size=(1400, 900))
display.start()

is_ipython = 'inline' in plt.get_backend()
if is_ipython:
    from IPython import display

plt.ion()