
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

# Environment

import gym
env = gym.make('FrozenLake-v0')

EPISODES = 2000
total_states = env.observation_space.n
actions_per_state = env.action_space.n
print('Number of states: ', total_states)
print('Number of actions per state: ', actions_per_state)
print('Number of episodes to run: ', EPISODES)



# Q-Table

q = np.zeros((total_states, actions_per_state), dtype='int')
Q = pd.DataFrame(q, columns=['Left', 'Down', 'Right', 'Up'])
Q.index = range(1,len(q)+1)
Q.reset_index(inplace=True)
Q.rename(columns={'index':'state'}, inplace=True)

# Helper functions to draw and update values of the table
def draw_Table(Q):
    table = ff.create_table(Q, index=True)
    table.layout.height, table.layout.width=350, 350
    for i in range(len(table.layout.annotations)):
        table.layout.annotations[i].align = 'center'
    return table

def update_Q(Q):
    for i in range(len(Q)):
        for j in range(1, len(Q.columns)):        
            Q.iloc[i,j] += np.random.choice([0,1,2])
    return Q



## Q-Learning Algorithm
#
#lr = .8
#y = .9
#rList = []
#
#for i in range(EPISODES):
#    
#    s = env.reset()
#    
#    rAll = 0
#    end_state = False
#    j = 0
#    
#    while j < 99:
#        
#        j+=1
#        
#        # Choose action greedily
#        a = np.argmax(Q[s,:] + np.random.randn(1, env.action_space.n) * (1./ (i+1)))
#        
#        # Collect reward and reach new state
#        s1,r,d,_ = env.step(a)
#        
#        # Update Q-Table with new knowledge
#        Q[s,a] = Q[s,a] + lr*(r + y*np.max(Q[s1,:]) - Q[s,a])
#        
#        rAll += r
#        s = s1
#        
#        if end_state == True: break
#
#    rList.append(rAll)
#


# Browser visualization

app.layout = html.Div([
        html.H1(children='Frozen Lake: Q-Learning Demo'),
        dcc.Graph(id='table', figure=draw_Table(Q)),
        dcc.Interval(
            id='time',
            interval=1*1000, # in milliseconds
            n_intervals=0)
        ]
    )

    
@app.callback(Output(component_id = 'table', component_property='figure'),
              [Input(component_id = 'time', component_property='n_intervals')])    
def update_table(n):   
    # Update values
    new_table = draw_Table(update_Q(Q))
    time.sleep(1 * np.exp(-n))
    return new_table
    

if __name__ == '__main__':
    app.run_server()