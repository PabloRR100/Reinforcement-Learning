
import time
import numpy as np
import pandas as pd
#import plotly.plotly as py
#import plotly.offline as pyo
import plotly.graph_objs as go
import plotly.figure_factory as ff

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

app = dash.Dash()
app.css.append_css({
    "external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"
})



# Environment

import gym
env = gym.make('FrozenLake-v0')

total_states = env.observation_space.n
actions_per_state = env.action_space.n
print('Number of states: ', total_states)
print('Number of actions per state: ', actions_per_state)



# Q - Table

q = np.zeros((total_states, actions_per_state), dtype='int')
Q = pd.DataFrame(q, columns=['Left', 'Down', 'Right', 'Up'])
Q.index = range(1,len(q)+1)
Q.reset_index(inplace=True)
Q.rename(columns={'index':'state'}, inplace=True)

table = ff.create_table(Q,  height_constant=20)
table.layout.width=300

def update_Q(Q):
    for i in range(len(Q)):
        for j in range(1, len(Q.columns)):        
            Q.iloc[i,j] += np.random.choice([0,1,2])
    print(Q)
    return Q

# Browser visualization

app.layout = html.Div([
        html.H1(children='Frozen Lake: Q-Learning Demo'),
        dcc.Graph(id='table', figure=table),
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
    new_table = ff.create_table(update_Q(Q))
    time.sleep(1)
    return new_table
    

if __name__ == '__main__':
    app.run_server()