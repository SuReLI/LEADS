import torch
import torch.nn as nn
import numpy as np
# from stable_baselines3.common.buffers import ReplayBuffer


def init_linear(input_dim, output_dim, requires_grad=True):
    layer = nn.Linear(input_dim, output_dim)
    for param in layer.parameters():
        param.requires_grad = requires_grad
    return layer

class RND(torch.nn.Module):
    def __init__(self,  state_dim):
        super(RND, self).__init__()
        self.l1_random = init_linear(state_dim, 256, requires_grad=False)
        self.l2_random = init_linear(256, 64, requires_grad=False)
        self.l3_random = init_linear(64, 1, requires_grad=False)
        # estimaor
        self.l1=torch.nn.Linear(state_dim, 256)
        self.l2=torch.nn.Linear(256, 64)
        self.l3=torch.nn.Linear(64, 1)
    
    def forward_random(self, s):
        x=torch.nn.functional.relu(self.l1_random(s))
        x=torch.nn.functional.relu(self.l2_random(x))
        x=torch.nn.functional.tanh(self.l3_random(x))
        return x
    
    def forward(self, s):
        x=torch.nn.functional.relu(self.l1(s))
        x=torch.nn.functional.relu(self.l2(x))
        x=torch.nn.functional.tanh(self.l3(x))
        return x

    def uncertainty_measure(self, s):
        with torch.no_grad():
            measure = (self.forward(s)-self.forward_random(s))**2
        return measure

    def loss(self,s): 
        return torch.mean((self.forward(s)-self.forward_random(s))**2)
        


