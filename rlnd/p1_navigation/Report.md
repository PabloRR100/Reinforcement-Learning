
# Summary of the Project

This projects creates a Deep RL Agent to solve the Banana Environment of Unity.
Two different kind of agents have been trained:
- DQN Agent
- Double DQN Agent

## Training definition
Both of them share the following properties:

- 2 Hidden Layers of 32 neurons each
- Learning Rate: 0.0005
- Batch Size: 64
- Discount Factor: 0.99
- Update Frequency of the Network: 4
- Soft Update of target parameters - Tau: 0.001
- Memory Capacity: 100.000

The training is defined by the following aspects:

- Max number of Episodes: 2000
- Max number of Time steps per Episode: 1000
- Initial epsilon (e-greedy): 1.0
- Minimal epsilon (minimal exploration): 0.01
- Epsilon decay rate: 0.995  


## Results 

The next figure represents the training process:
![image][training]

## Comments
- Why the Double DQN Agent performs worse than the simple DQN Agent?

## Future Steps
- **Implement a Prioritized Experience Replay.**  
Give a higher probability of being sampled from the memory to those tuples that had a big target error while training  
- **Implement a Dueling DQN**  
Have a single network that produced two outputs, a value function estimation and an advantage estimation, to combine both when predicting the Q-value.

[training]: 'training.png'
