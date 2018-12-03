[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"


# Introduction

This repository is the first project of the Deep Reinforcement Learning Nanodegree Program at **[Udacity Deep Reinforcement Learning Nanodegree][udacity]**.

In the second part of the Nanodegree we have studied **Value-Based Methods** including **Deep Q-Learning**, Double DQN, Prioritized Replay and Dueling DQN. 

The project goal is to use Unity ML-Agents([github repository](https://github.com/Unity-Technologies/ml-agents)) to create an agent that navigates a world picking yellow bananas and avoid picking blue bananas.  


# Project
![Trained Agent][image1]

In the simulator, there are two types of bananas, yellow ones and blue ones.  
The yellow ones give positive reward (+1) but blue ones give negative reward (-1). Thus, the goal of the agent is to collect
as many yellow bananas as possible while avoiding blue bananas.  

The action space allows the agent to perform one out of 4 actions at every time step:
- `0` - walk forward
- `1` - walk backward
- `2` - turn left
- `3` - turn right

The first implementation is not directly from row pixels. Instead, the state space has `37` dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.


# Getting Started
1. Check [this nanodegree's prerequisite](https://github.com/udacity/deep-reinforcement-learning/#dependencies), and follow the instructions.

2. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

3. Place the file in `bin/` directory, and unzip (or decompress) the file.


# Instructions
To train the agent, start jupyter notebook, open `Navigation_Submission.ipynb` inside the notebook.
There will be a first part corresponding to training a DQN Agent and the second part for the Double DQN Agent.

### Organization
- navigation.py contains the same code as the notebook to be run as a single script
- dqn.py contains the code for the DQN and DoubleDQN Agents and the Memory Replay Implementations

### Trained Agents
The trained agents can be found in the folder ```models_backups```

### Future updates
- Implement a prioritized experience replay
- Implement a Dueling network architecture


# Final Report
A summary of the project and the performance report can be found in the ```Report.ipynb``` notebook.

[udacity]: https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893