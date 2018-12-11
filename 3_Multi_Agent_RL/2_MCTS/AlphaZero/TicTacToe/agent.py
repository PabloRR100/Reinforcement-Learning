# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class Agent(nn.Module):

    def __init__(self):
        super(Agent, self).__init__()
        
        self.conv = nn.Conv2d(1, 16, kernel_size=2, stride=1, bias=False)
        self.size = 2*2*16
        self.fc = nn.Linear(self.size,32)

        # Layers for the policy
        self.fc_action1 = nn.Linear(32, 16)
        self.fc_action2 = nn.Linear(16, 9)
        
        # Layers for the critic
        self.fc_value1 = nn.Linear(32, 8)
        self.fc_value2 = nn.Linear(8, 1)
        
        # Activations
        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyRelu()
        self.tanh = nn.Tanh()
        
    def forward(self, x):

        y = self.relu(self.conv(x))
        y = y.view(-1, self.size)
        y = self.relu(self.fc(y))
                
        # the action head
        a = self.relu(self.fc_action1(y))
        a = self.fc_action2(a)
        # availability of moves
        avail = (torch.abs(x.squeeze())!=1).type(torch.FloatTensor)
        avail = avail.view(-1, 9)
        
        # Set the prob to zero where actions are not possible
        maxa = torch.max(a)
        # subtract off max for numerical stability (avoids blowing up at infinity)
        exp = avail*torch.exp(a-maxa)
        prob = exp/torch.sum(exp)
        
        # The value head
        value = self.relu(self.fc_value1(y))
        value = self.tanh(self.fc_value2(value))
        return prob.view(3,3), value
