
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

'''
Intuitive video about sparse rewards:
    <https://www.youtube.com/watch?v=0Ey02HT_1Ho>
'''

# Environment

import gym
env = gym.make('FrozenLake-v0')

EPISODES = 20
total_states = env.observation_space.n
actions_per_state = env.action_space.n
print('Number of states: ', total_states)
print('Number of actions per state: ', actions_per_state)
print('Number of episodes to run: ', EPISODES)

actions = ['Left', 'Down', 'Right', 'Up']
states = ['S',' ',' ',' ', 'F','H','F','H','F','F','F','H','H','F','F','G']
world = [['S',' ',' ',' '],
         [' ',' ',' ','H'],
         [' ',' ',' ','H'],
         ['H',' ',' ','G']]

# Helper function to track movement
def draw_world(w, s):
    i = int(np.floor(s / 4))
    j = s % 4
    w[i,j] = 'X'
    print(w)


# Q-Table

q = np.zeros((total_states, actions_per_state), dtype='float')
Q = pd.DataFrame(q, columns=actions)
Q.reset_index(inplace=True)
Q.rename(columns={'index':'State'}, inplace=True)
Q.set_index('State', inplace=True)

# Helper functions to draw, update and get values of the table
def draw_Table(Q):
    table = sns.heatmap(Q, cmap='Blues', annot=True, linewidths=.5, cbar=False, 
                linecolor='black', square=True).set_title('Q-Table')
    return table  

def set_Q(q): 
    global Q
    Q.iloc[:,:] = q



# Q-Learning Algorithm

lr = .8
y = .9
rList = []

    
plt.ion()
plt.figure(figsize = (10,10))
for i in range(EPISODES):
    
    print('Episode [{}/{}]'.format(i,EPISODES))
    print('Current Q-Table')

    # if i == 0: time.sleep(5) # time to plot the stating Q-Table
    s = env.reset()
    
    rAll = 0
    end_state = False
    j = 0
    
    while j < 10:
        
        j+=1
        
        # Choose action greedily
        a = np.argmax(q[s,:] + np.random.randn(1, env.action_space.n) * (1./ (i+1)))
        
        # Collect reward and reach new state
        s1,r,end_state,inf = env.step(a)
        # print(actions[a], s1, states[s1],r)
        
        # Encourage a little bit for testing
        r += 1
        
        # Update Q-Table with new knowledge
        q[s,a] = round(q[s,a] + lr*(r + y*np.max(q[s1,:]) - q[s,a]), 2)
        
        set_Q(q)
        
        # Update the new Q-Value
        if 'previous' in globals(): del previous
        previous = draw_Table(Q)
        plt.pause(1)
        
        rAll += r
        s = s1
        
        if end_state == True: 
            a = 5
            break
    
    rList.append(rAll)

plt.ioff()
plt.show()


## Browser visualization
#
#def update_table(n):   
#    # Update values
#    new_table = draw_Table(get_Q())
##    time.sleep(m * np.exp(-n))
#    return new_table
#    
    