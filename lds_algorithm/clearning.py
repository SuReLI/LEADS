import torch
import torch.nn as nn
import numpy as np
import random
import time
from utils.tools import make_gradient_clip_hook, gradient_clip_hook


class MeasureNetwork(torch.nn.Module):
    def __init__(self,  state_dim, action_dim, z_dim, env_name, featurize = False):
        super(MeasureNetwork, self).__init__()
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
        elif self.env_name == 'half_cheetah':
            self.feature_dim = 3
        elif self.env_name == 'hand':
            self.feature_dim = 15
        self.l1=torch.nn.Linear(state_dim+action_dim+state_dim+z_dim, 256) if not self.featurize else torch.nn.Linear(state_dim+action_dim+self.feature_dim+z_dim, 256)
        self.l2=torch.nn.Linear(256, 1)
        
    def forward(self, s1, a1, s2, z):
        x=torch.cat([s1, a1, s2, z], dim=1) if not self.featurize else torch.cat([s1, a1, self.featurizer(s2), z], dim=1)
        x=torch.nn.functional.relu(self.l1(x))
        x=self.l2(x)
        x=torch.sigmoid(x)*(1-1e-3)
        return x

    def featurizer(self, s):
        if self.env_name == 'hopper':
            return s[: , -3:]
        elif self.env_name == 'fetch':
            return s[: , 0:3]
        elif self.env_name == 'fetch_push':
            return s[: , 3:5]
        elif self.env_name == 'finger':
            return s[: , 0:3]
        elif self.env_name == 'half_cheetah':
            return s[: , -3:]
        elif self.env_name == 'hand':
            return s[: , 48:]
        

class Clearning:
    def __init__(self, state_dim, action_dim, z_dim, learning_rate=0.001, device='cpu', gamma=0.99, _lambda = 0.2, env_name = 'Easy', featurizer = False ):
        self.env_name = env_name
        self.C = MeasureNetwork(state_dim, action_dim, z_dim, self.env_name, featurizer).to(device)
        self.optimizer = torch.optim.Adam(self.C.parameters(), lr=learning_rate)
        self.loss_fn = nn.BCELoss(reduction='none')
        self.gamma=gamma
        self.device = device
        self._lambda=_lambda
        self.tar= 1e-9
        self.hooks=[]

        # Monte-Carlo C-learning
    def loss_mc(self, s_current, a_current, s_future, s_random, z):
        obj_pos = torch.log(self.C(s_current, a_current, s_future, z)+self.tar).mean()
        obj_neg = torch.log(1-self.C(s_current, a_current, s_random, z)+self.tar).mean()
        loss = - (obj_pos + obj_neg)
        return loss

    # Temporal Difference C-learning
    def loss_td(self, s_current, a_current, s_next, a_next, s_random, z):
        obj_a = (1 - self.gamma) * torch.log(self.C(s_current, a_current, s_next, z)+self.tar)
        obj_b = torch.log(1 - self.C(s_current, a_current, s_random, z)+self.tar)
        with torch.no_grad() : 
            w = self.C(s_next, a_next, s_random, z) / (1 - self.C(s_next, a_next, s_random, z)+self.tar)
            # print('w mean : ', self.C(s_next, a_next, s_random, z).mean(), ' w max : ', self.C(s_next, a_next, s_random, z).max(), ' w min : ', self.C(s_next, a_next, s_random, z).min())
            w = torch.clamp(w, 0, 20) 
        obj_c = self.gamma * w * torch.log(self.C(s_current, a_current, s_random, z)+self.tar)
        objective = obj_a + obj_b + obj_c
        return -1.0 * torch.mean(objective)
    
    def update_vector(self,rb_states,rb_actions,rb_z, codebook, actor, batch_size, n_episode, lambda_c):
        i_0 = rb_states.shape[0]
        i_1 = rb_states.shape[1]
        i_2 = rb_states.shape[2]
        i_3 = rb_states.shape[3]
        # mc
        mc_idx_z = np.random.randint(0,i_0, batch_size)
        mc_idx_episode = np.random.randint(i_1-n_episode,i_1, batch_size)
        mc_idx_s = np.random.randint(0,i_2-1, batch_size)
        # a
        mc_idx_a = mc_idx_s
        # future
        diff = i_2 - mc_idx_s
        mc_idx_future = np.round(np.random.rand(batch_size) * diff + mc_idx_s).astype(int)-1
        # random
        mc_idx_z_random = np.random.randint(0,i_0, batch_size)
        mc_idx_episode_random = np.random.randint(0,i_1, batch_size)
        mc_idx_s_random = np.random.randint(0,i_2, batch_size)
        # mc batch 
        z_mc = torch.from_numpy(rb_z[mc_idx_z, mc_idx_episode]).to(self.device)
        s_mc = torch.from_numpy(rb_states[mc_idx_z, mc_idx_episode, mc_idx_s].astype(np.float32)).to(self.device)
        s_next_mc = torch.from_numpy(rb_states[mc_idx_z, mc_idx_episode, mc_idx_s+1].astype(np.float32)).to(self.device)
        a_mc = torch.from_numpy(rb_actions[mc_idx_z, mc_idx_episode, mc_idx_a]).to(self.device)
        s_future_mc = torch.from_numpy(rb_states[mc_idx_z, mc_idx_episode, mc_idx_future].astype(np.float32)).to(self.device)
        s_random_mc = torch.from_numpy(rb_states[mc_idx_z_random, mc_idx_episode_random, mc_idx_s_random].astype(np.float32)).to(self.device)
        # td 
        td_idx_z = np.random.randint(0,i_0, batch_size)
        td_idx_episode = np.random.randint(0,i_1, batch_size)
        td_idx_s = np.random.randint(0,i_2-1, batch_size)
        td_idx_a = td_idx_s
        td_idx_s_next = td_idx_s + 1
        td_idx_z_random = np.random.randint(0,i_0, batch_size)
        td_idx_episode_random = np.random.randint(0,i_1, batch_size)
        td_idx_s_random = np.random.randint(0,i_2, batch_size)
        # td batch 
        # Maze
        z_td = torch.from_numpy(rb_z[td_idx_z, td_idx_episode]).to(self.device)
        s_td = torch.from_numpy(rb_states[td_idx_z, td_idx_episode, td_idx_s].astype(np.float32)).to(self.device)
        a_td = torch.from_numpy(rb_actions[td_idx_z, td_idx_episode, td_idx_a]).to(self.device)
        s_next_td = torch.from_numpy(rb_states[td_idx_z, td_idx_episode, td_idx_s_next].astype(np.float32)).to(self.device)
        s_random_td = torch.from_numpy(rb_states[td_idx_z_random, td_idx_episode_random, td_idx_s_random].astype(np.float32)).to(self.device)
        
        with torch.no_grad():
            a_next_td, _, _ = actor.get_action(s_next_td, z_td)
        # mc loss
        loss_mc = self.loss_mc(s_mc, a_mc, s_future_mc, s_random_mc, z_mc)
        # td loss
        loss_td = self.loss_td(s_td, a_td, s_next_td, a_next_td, s_random_td, z_td)
        # total loss
        loss = lambda_c*loss_mc + (1-lambda_c)*loss_td
        # update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss_mc.item(), loss_td.item()

     

    def measure(self, s1, a1, s2, z):
        F1=self.C(s1,a1,s2,z)
        m = F1/(1-F1)
        # self.hooks.append(m.register_hook(gradient_clip_hook)) if m.grad != None else None
        return m
        # return F1
   
    def measure_bij(self,s1,a1,s2,z):
        return self.C(s1,a1,s2,z)

