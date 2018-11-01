

import pong_utils
device = pong_utils.device
print("using device: ", device)

import gym
import time
env = gym.make('PongDeterministic-v4')
# PongDeterministic does not contain random frameskip
# so is faster to train than the vanilla Pong-v4 environment

print("List of available actions: ", env.unwrapped.get_action_meanings())
# The actions 'RIGHTFIRE' = 4 and 'LEFTFIRE" = 5 makes the game restarts if done

import matplotlib
import matplotlib.pyplot as plt

from agent import Agent
agent = Agent()
agent = agent.to(device)

pong_utils.play(env, agent, time=100) 

envs = pong_utils.parallelEnv('PongDeterministic-v4', n=4, seed=12345)
prob, state, action, reward = pong_utils.collect_trajectories(envs, agent, tmax=100)
    
    
    