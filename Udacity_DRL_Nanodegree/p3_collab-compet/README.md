[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"


# Project 3: Collaboration and Competition

### Introduction

This project solves the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment of Unity.

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the file in the DRLND GitHub repository, in the `p3_collab-compet/` folder, and unzip (or decompress) the file. 


### Project Organization

The instructions to solve the environment can be found in `Tennis_Submission.ipynb`.  
In order to create your own agent, you should modify the files ```networks.py``` or ```agent.py```.  

#### ```Networks.py```
In this script you will find the neural networks (in PyTorch) used as the brains of the Actor and the Critic.  
Feel free to modify them to increment their capacity. Note that bigger networks will probably have more ability to learn but at the cost of more training time, since there are more parameters to adjust (or learn).  

#### ```Agent.py```
In this script you will find the Agent class.  
This class will import the Actor and Critic brains to the agent.  
Also, there ReplayBuffer class will be instantiated as the agent's memory. Try also modifying the hyperparamters of this memory.   

The agents have the ability to:  
 - Act -> Given a policy and a state, returns an action  
 - Step -> Add an experience (tuple) to the memory
 - SampleandLearn -> Use a random sampling from that memory to call learn function
 - Learn -> Update the Actor and Critic == Update the Policy and the Value Parameters
 - Soft Target -> Soft update to melt the target parameters to the regural parameters

### Results
The results obtained during the training and the metrics of performance can be found in the [report][1] attached.  
The saved models (Checkpoints) can be found in the [model_backups][2] folder.

[1]: https://github.com/PabloRR100/Reinforcement-Learning/tree/master/Udacity_DRL_Nanodegree/p3_collab-compet/report
[2]: https://github.com/PabloRR100/Reinforcement-Learning/tree/master/Udacity_DRL_Nanodegree/p3_collab-compet/model_backups
