
# Continuous Control Project


## Algorithm: Deep Deterministic Policy Gradient
The RL algorithm used to solve this project is Deep Deterministic Policy Gradient, which was introduced by DeepMind in the publication ["Continuous control with deep reinforcement learning"][paper].  

DDPG solves the problem of having a continuous action space.  
DDPG uses the Actor-Critic framework, but it is considered by some researchers as an extension of DQN since uses the target network key idea. The critic is used to approximate the maximizer over the *q-values* of the next state, instead of being used as a learned baseline.   

##### Actor
Is used to approximate the optimal policy ***deterministically***. It always outputs the best believed action. Stochastic methods on the other hand want to learn a probability distribution over actions.  
This actor is learning the *argmaxQ(s,a)* action.

##### Critic
It does the policy evaluation step. Therefore, it learns to evaluate the optimal action-value funtion by using the actors best believed actions.  

##### Replay Buffer
DDPG uses a Replay Buffer to decorrelate the tuples of experiences obtained sequentially.  
However, the uptades are slightly different than how the Replay Buffer works on other algorithms like DQN.  

This time, they are called ***soft updates***. 
This strategy consists on *slowly blending* the Regular Network weights into the Target weigths. Every timestep, the Target is 99% weights of its own and 1% the weights of the Regular Network.

This is opposed to the DQN implementation where the Target Network is updated every 10000 steps by coping the weights of the Regular Network into the Target Network. The Target Network then is fixed for a long time and then it sufferts a big update, intead of the slow blending mentioned above.



## Training Process

```python
# Init Training
# -------------

agent, scores = train(ENV)
results = dict(agent=agent, scores=scores)
```

    Loading environmnet...
    


    INFO:unityagents:
    'Academy' started successfully!
    Unity Academy name: Academy
            Number of Brains: 1
            Number of External Brains : 1
            Lesson number : 0
            Reset Parameters :
    		goal_speed -> 1.0
    		goal_size -> 5.0
    Unity brain name: ReacherBrain
            Number of Visual Observations (per agent): 0
            Vector Observation space type: continuous
            Vector Observation space size (per agent): 33
            Number of stacked Vector Observation: 1
            Vector Action space type: continuous
            Vector Action space size (per agent): 4
            Vector Action descriptions: , , , 


    Loading agenst...
    
    Number of Agents loaded:  20
    Capacity of the Actor (# of parameters):  162436
    Capacity of the Critic (# of parameters):  108545
    Initializing training...
    
    Episode 1, Average last 100 scores: 0.01, Episode Duration: 561.76, 
    
    Episode 2, Average last 100 scores: 0.01, Episode Duration: 621.55, 
    
    Episode 3, Average last 100 scores: 0.01, Episode Duration: 631.52, 
    
    Episode 4, Average last 100 scores: 0.01, Episode Duration: 648.75, 
    
    Episode 5, Average last 100 scores: 0.01, Episode Duration: 655.82, 
    
    Episode 6, Average last 100 scores: 0.01, Episode Duration: 674.15, 
    
    Step 500	Average Score: 0.02

**I stopped the training here since it was not improving at all and I only have 4 GPU hours left for finish this project and next one**

## Results 

```python
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
```

This code will plot the evolution of the scores once the training is completed
![png](output_9_1.png)


## Future Improvements  

    1 - Tuning of the hyperparameters to speed up training and increase performance.  
    2 - Try different algorithms:
      [Continuous Deep Q-Learning with Model-based Acceleration][2] 

[paper]: https://arxiv.org/abs/1509.02971
[2]: https://arxiv.org/abs/1603.00748
