import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class SoftQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, device='cpu'):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim , 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 3
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_space):
        super().__init__()
        self.fc1 = nn.Linear(state_dim , 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, action_dim)
        self.fc_logstd = nn.Linear(256, action_dim)
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((action_space.high - action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((action_space.high + action_space.low) / 2.0, dtype=torch.float32)
        )
        self.hooks=[]

    def forward(self, x ):
        # x = torch.cat([x, z], 1)
        x = F.relu(self.fc1(x)) 
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        # mean = torch.tanh(mean)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats
        # print(torch.max(mean)) if torch.max(mean) > 3 else None
        return mean, log_std

    def get_action(self, x ):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std) 
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        log_prob = log_prob 
        return action, log_prob, mean

    def entropy(self, x, z):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        entropy = normal.entropy()
        return entropy.sum(1, keepdim=True)
    
    # def get_action_probabilities(self, state, num_samples=10,device='cpu'):
    #     mean, log_std = self(state)
    #     mean = mean
    #     std = log_std.exp()
    #     std = std
    #     low = (self.action_scale*-1) + self.action_bias
    #     high = self.action_scale + self.action_bias
    #     # Generate num_samples action values, centered around mean, covering a range of std deviations
    #     actions = torch.randn(size=(*mean.shape,num_samples), device=mean.device) * std.unsqueeze(-1) + mean.unsqueeze(-1)
    #     # Clip the actions to the valid action space range
    #     # low = low.unsqueeze(0).unsqueeze(2)
    #     # high = high.unsqueeze(0).unsqueeze(2)
    #     # a_clamped = actions.clamp(min=low, max=high)
    #     # Calculate the probabilities of these actions under the actor's policy
    #     normal_distribution = torch.distributions.Normal(mean.unsqueeze(-1), std.unsqueeze(-1))
    #     action_probabilities = normal_distribution.log_prob(actions).exp()
    #     action_probabilities = action_probabilities.prod(dim=1)
    #     # Actions outside the valid range get a probability of 0
    #     # action_probabilities *= (actions >= low) * (actions <= high)

    #     return actions, action_probabilities