
import time
import numpy as np
import pandas as pd
#import plotly.plotly as py
#import plotly.offline as pyo
#import plotly.graph_objs as go
import plotly.figure_factory as ff

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

app = dash.Dash()

'''
Intuitive video about sparse rewards:
    <https://www.youtube.com/watch?v=0Ey02HT_1Ho>
'''



# Environment

import gym
env = gym.make('FrozenLake-v0')

EPISODES = 2
total_states = env.observation_space.n
actions_per_state = env.action_space.n
print('Number of states: ', total_states)
print('Number of actions per state: ', actions_per_state)
print('Number of episodes to run: ', EPISODES)

actions = ['Left', 'Down', 'Right', 'Up']
states = ['S','F','F','F', 'F','H','F','H','F','F','F','H','H','F','F','G']
world = [['S','F','F','F'],
         ['F','H','F','H'],
         ['F','F','F','H'],
         ['H','F','F','G']]



# Q-Table

q = np.zeros((total_states, actions_per_state), dtype='float')
Q = pd.DataFrame(q, columns=actions)
Q.index = range(1,len(q)+1)
Q.reset_index(inplace=True)
Q.rename(columns={'index':'State'}, inplace=True)
Q.set_index('State', inplace=True)

# Helper functions to draw and update values of the table
def draw_Table(Q):
    table = ff.create_table(Q, index=True)
    table.layout.height, table.layout.width=350, 350
    for i in range(len(table.layout.annotations)):
        table.layout.annotations[i].align = 'center'
    return table

def set_Q(q): 
    global Q
    Q.iloc[:,:] = q
    
def get_Q():
    global Q
    return Q

def update_Q(Q):
    for i in range(len(Q)):
        for j in range(1, len(Q.columns)):        
            Q.iloc[i,j] += np.random.choice([0,1,2])
    print(Q)
    return Q


# Browser visualization

app.layout = html.Div([
        html.H1(children='Frozen Lake: Q-Learning Demo'),
        html.P(id='placeholder'),
        dcc.Graph(id='table', figure=draw_Table(Q)),
        #dcc.Interval(id='time', interval=1*10e6, n_intervals=0),
        dcc.Interval(id='time2', interval=1*1000, n_intervals=0),
        ]
    )

#
## Q-Learning Algorithm
#@app.callback(Output(component_id = 'placeholder', component_property='hidden'),
#              [Input(component_id = 'time', component_property='n_intervals')])    
#def algorithm(n):
#    
#    global q
#    global Q
#    lr = .8
#    y = .9
#    rList = []
#    
#    print('Running episodes')
#    for i in range(EPISODES):
#    
#        print('Episode [{}/{}]'.format(i,EPISODES))
#        print('Current Q-Table')
#        print(Q)
#        
#        # if i == 0: time.sleep(5) # time to plot the stating Q-Table
#        s = env.reset()
#        
#        rAll = 0
#        end_state = False
#        j = 0
#        
#        while j < 99:
#            
#            j+=1
#            
#            time.sleep(1 * np.exp(-j))
#            
#            # Choose action greedily
#            a = np.argmax(q[s,:] + np.random.randn(1, env.action_space.n) * (1./ (i+1)))
#            
#            # Collect reward and reach new state
#            s1,r,end_state,inf = env.step(a)
#            # print(actions[a], s1, states[s1],r)
#            
#            # Encourage a little bit for testing
#            r += 1
#            
#            # Update Q-Table with new knowledge
#            q[s,a] = round(q[s,a] + lr*(r + y*np.max(q[s1,:]) - q[s,a]), 2)
#            # print(inf)
#            
#            set_Q(q)
#            
#            rAll += r
#            s = s1
#            
#            if end_state == True: 
#                a = 5
#                break
#    
#        rList.append(rAll)
        

# Timer for the Visualization Update
@app.callback(Output(component_id = 'table', component_property='figure'),
              [Input(component_id = 'time2', component_property='n_intervals')])    
def update_table(n):   
    # Update values
    global Q
    print('Updating table ')
#    new_table = draw_Table(Q)
    new_table = draw_Table(update_Q(Q))
    #time.sleep(1 * np.exp(-n))
    time.sleep(1)
    return new_table
    


if __name__ == '__main__':
    app.run_server(debug=True)