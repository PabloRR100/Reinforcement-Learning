
# Torch Implementation
import torch.nn as nn

class Agent(nn.Module):
    
    def __init__(self):
        super(Agent, self).__init__()
        
        # outputsize = (inputsize - kernel_size + stride)/stride 
        # Let's start with a simple network. More complex one commented out

        # output = 20x20 here
        self.conv1 = nn.Conv2d(2, 4, kernel_size=2, stride=2)  ## 80x80x2 --> 40x40x4
        self.conv2 = nn.Conv2d(4, 16, kernel_size=2, stride=2) ## 40x40x4 --> 20x20x16 
        
        self.size = 20*20*16 # 6400
        
        # 1 fully connected layer        
        self.fc1 = nn.Linear(self.size, 200)
        self.fc2 = nn.Linear(200, 1)
        self.drop = nn.Dropout(inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.sig = nn.Sigmoid()
        
    def forward(self, x):
        
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(-1,self.size)
        x = self.fc2(self.fc1(x))
        return self.sig(x) # sigmoid to squeeze logp [0-1]
    