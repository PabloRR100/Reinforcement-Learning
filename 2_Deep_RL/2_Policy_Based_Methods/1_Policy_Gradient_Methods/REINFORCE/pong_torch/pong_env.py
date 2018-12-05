

import pong_utils
device = pong_utils.device
print("using device: ", device)

import gym
env = gym.make('PongDeterministic-v4')
print("List of available actions: ", env.unwrapped.get_action_meanings())
# The actions 'RIGHTFIRE' = 4 and 'LEFTFIRE" = 5 makes the game restarts if done

import matplotlib.pyplot as plt

from agent import Policy
agent = Policy()
agent = agent.to(device)

pong_utils.play(env, agent, time=100) 

envs = pong_utils.parallelEnv('PongDeterministic-v4', n=4, seed=12345)
prob, state, action, reward = pong_utils.collect_trajectories(envs, agent, tmax=100)
    
    
    