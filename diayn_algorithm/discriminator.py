import torch
import torch.nn as nn
import numpy as np


class Discriminator(torch.nn.Module):
    def __init__(self,  state_dim, n_skill, env_name, featurize):
        super(Discriminator, self).__init__()
        self.env_name = env_name
        self.featurize = featurize
        if self.env_name == 'hopper':
            self.feature_dim = 3
        elif self.env_name == 'fetch':
            self.feature_dim = 3
        elif self.env_name == 'fetch_push':
            self.feature_dim = 2
        elif self.env_name == 'finger':
            self.feature_dim = 3
        self.l1=torch.nn.Linear(state_dim, 256) if not self.featurize else torch.nn.Linear(self.feature_dim, 256)
        # self.l1=torch.nn.Linear(state_dim, 256)
        self.l2=torch.nn.Linear(256, 64)
        self.l3=torch.nn.Linear(64, n_skill)
    
    def forward(self, s):
        x = self.featurizer(s) if self.featurize else s
        x=torch.nn.functional.relu(self.l1(x))
        x=torch.nn.functional.relu(self.l2(x))
        x=self.l3(x)
        x=torch.softmax(x, dim=1)
        return x

    def featurizer(self, s):
        if self.env_name == 'hopper':
            return s[: , -6:-3]
        elif self.env_name == 'fetch':
            return s[: , 0:3]
        elif self.env_name == 'fetch_push':
            return s[: , 3:5]
        elif self.env_name == 'finger':
            return s[: , 0:3]