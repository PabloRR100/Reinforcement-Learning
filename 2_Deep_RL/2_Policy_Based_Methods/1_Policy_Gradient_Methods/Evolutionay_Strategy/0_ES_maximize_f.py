
"""
A bare bones examples of optimizing a black-box function (f) using
Natural Evolution Strategies (NES), where the parameter distribution is a 
gaussian of fixed standard deviation.
"""

import numpy as np
np.random.seed(0)

# Solution to find
solution = np.array([0.5, 0.1, -0.3])

def f(w):
  # Minimize the L2 distance to a specific solution vector (0 is the max reward)
  reward = -np.sum(np.square(solution - w))
  return reward

# Hyperparameters
npop = 50 # population size
sigma = 0.1 # noise standard deviation
alpha = 0.001 # learning rate
episodes = 300 # episodes to run the task


# Initial guess is random
w = np.random.randn(3)

for i in range(episodes):

  # Print current fitness of the most likely parameter setting
  if i % 20 == 0:
    print('iter %d. w: %s, solution: %s, reward: %f' % 
          (i, str(w), str(solution), f(w)))

  # Initialize memory for a population of w's, and their rewards
  N = np.random.randn(npop, 3) 
  R = np.zeros(npop)
  
  # For each wieght vector in the population vector
  for j in range(npop):
    
    # Jitter w injecting gaussian noise
    w_try = w + sigma*N[j] 
    # Tvaluate
    R[j] = f(w_try)

  # Standardize the rewards to have a gaussian distribution
  A = (R - np.mean(R)) / np.std(R)
  # Update rule. Weighted (by reward obtain) sum of rewards of the npop agents
  w = w + alpha/(npop*sigma) * np.dot(N.T, A)

