import torch
import torch.nn as nn
import numpy as np


class Discriminator(torch.nn.Module):
    def __init__(self,  state_dim, z_dim, env_name, featurize):
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
        self.l2=torch.nn.utils.parametrizations.spectral_norm(torch.nn.Linear(256, 64))
        self.l3=torch.nn.utils.parametrizations.spectral_norm(torch.nn.Linear(64, z_dim))
        # self.l1=torch.nn.Linear(state_dim, 256)
        # self.l2=torch.nn.Linear(256, 64)
        # self.l3=torch.nn.Linear(64, z_dim)
    
    def forward(self, s):
        x = self.featurizer(s) if self.featurize else s
        x=torch.nn.functional.relu(self.l1(x))
        x=torch.nn.functional.relu(self.l2(x))
        x=self.l3(x)
        # x=torch.sigmoid(self.l3(x))
        # x=torch.softmax(x, dim=1)
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

# import torch
# import torch.nn as nn

# class VariableLengthLSTM(nn.Module):
#     def __init__(self, state_space_dim, hidden_dim, num_layers=1):
#         super(VariableLengthLSTM, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers
#         # LSTM Layer
#         self.lstm = nn.LSTM(state_space_dim, hidden_dim, num_layers, batch_first=True)
#         # Fully connected layer
#         self.fc = nn.Linear(hidden_dim, state_space_dim)

#     def forward(self, x):
#         # x is a list of tensors with variable length, shape of each tensor: (sequence_length, state_space_dim)
#         # Packing the sequence
#         x_packed = nn.utils.rnn.pack_sequence(x, enforce_sorted=False)

#         # LSTM forward
#         packed_output, (hidden, cell) = self.lstm(x_packed)

#         # Unpack the sequence
#         output, input_sizes = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

#         # We use the last hidden state to pass through the fully connected layer
#         out = self.fc(hidden[-1])

#         return out

# if __name__=='__main__': 
#     discriminator = VariableLengthLSTM(10, 5, 3)
