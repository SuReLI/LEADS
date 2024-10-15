import torch
import torch.nn as nn
import numpy as np
import random
from env.maze.maze import Maze
from env.mujoco.wmujoco import WMujoco
import time


class MeasureNetwork(torch.nn.Module):
    def __init__(self,  mujoco, state_dim, z_dim, action_dim, feature_dim = 2):
        super(MeasureNetwork, self).__init__()
        self.mujoco = mujoco
        # self.mujoco = False
        self.mujoco = True
        self.l1=torch.nn.Linear(state_dim+action_dim+feature_dim+z_dim, 256) 
        self.l2=torch.nn.Linear(256, 32)
        self.l3=torch.nn.Linear(32, 1)
        # self.l4=torch.nn.Linear(32, 1)
    
    def forward(self, s1, a1, s2, z):
        x=torch.cat([s1, a1, self.featurizer(s2), z], dim=1) 
        x=torch.nn.functional.relu(self.l1(x))
        x=torch.nn.functional.relu(self.l2(x))
        x=self.l3(x)
        # x=torch.nn.functional.relu(self.l4(x))
        x=torch.sigmoid(x)*10
        return x

    def featurizer(self, s) : 
        return s[:, 0:2]


class Clearning:
    def __init__(self, env_class, state_dim, action_dim, z_dim, learning_rate=0.001, device='cpu', gamma=0.99, _lambda = None, sigma = 0.1 , tau = 0.005):

        self.mujoco = True if isinstance(env_class(), WMujoco) else False
        # self.C = MeasureNetworkFactorized(self.mujoco, state_dim, action_dim, z_dim).to(device) if isinstance(env_class(), WMujoco) else MeasureNetwork(self.mujoco, state_dim, action_dim, z_dim).to(device)
        self.C = MeasureNetwork(self.mujoco, state_dim, z_dim, action_dim).to(device)
        self.C_target = MeasureNetwork(self.mujoco, state_dim, z_dim, action_dim).to(device)
        self.C_target.load_state_dict(self.C.state_dict())
        self.optimizer = torch.optim.Adam(self.C.parameters(), lr=learning_rate)
        self.loss_fn = nn.BCELoss(reduction='none')
        self.mse_loss = torch.nn.MSELoss()
        self.gamma=gamma
        self.sigma = sigma 
        self.tau = tau
        self.device = device
        self.tar= 1e-9
        self.hooks=[]
        

    def loss_td(self, s, a, s_next, a_next, s_e, z):
        with torch.no_grad():
            phi_next  = self.C.featurizer(s_next)
            normal_next = torch.distributions.Normal(phi_next, self.sigma* torch.ones(phi_next.shape[0],phi_next.shape[1], device = self.device))
            phi_next_tilde = normal_next.rsample()
            phi_future = self.C.featurizer(s_e)
            phi_e = (1-self.gamma)*phi_next_tilde + self.gamma*phi_future
            p_f = torch.sum(torch.log(torch.exp(normal_next.log_prob(phi_e))*(1-self.gamma) + self.gamma*self.C_target(s_next,a_next,s_e,z)+1e-9), dim =1, keepdim = True)
        loss = self.mse_loss(torch.log(self.C(s,a,s_e,z)+1e-9), p_f)
        return loss

    def exponential_moving_average(self):
        for target_param, source_param in zip(self.C_target.parameters(), self.C.parameters()):
            target_param.data = self.tau * target_param.data + (1 - self.tau) * source_param.data
    
    def update_vector(self,rb_states,rb_actions,rb_z, codebook, actor, batch_size, n_episode, lambda_c):
        i_0 = rb_states.shape[0]
        i_1 = rb_states.shape[1]
        i_2 = rb_states.shape[2]
        i_3 = rb_states.shape[3]
        # mc
        mc_idx_z = np.random.randint(0,i_0, batch_size)
        mc_idx_episode = np.random.randint(i_1-n_episode,i_1, batch_size)
        mc_idx_s = np.random.randint(0,i_2-1, batch_size)
        mc_idx_a = mc_idx_s
        diff = i_2 - mc_idx_s
        mc_idx_future = np.round(np.random.rand(batch_size) * diff + mc_idx_s).astype(int)-1
        z_mc = torch.from_numpy(rb_z[mc_idx_z, mc_idx_episode]).to(self.device)
        s_mc = torch.from_numpy(rb_states[mc_idx_z, mc_idx_episode, mc_idx_s]).to(self.device)
        a_mc = torch.from_numpy(rb_actions[mc_idx_z, mc_idx_episode, mc_idx_a]).to(self.device)
        s_next_mc = torch.from_numpy(rb_states[mc_idx_z, mc_idx_episode, mc_idx_s + 1]).to(self.device)
        s_future_mc = torch.from_numpy(rb_states[mc_idx_z, mc_idx_episode, mc_idx_future]).to(self.device)
        # td 
        td_idx_z = np.random.randint(0,i_0, batch_size)
        td_idx_episode = np.random.randint(0,i_1, batch_size)
        td_idx_s = np.random.randint(0,i_2-1, batch_size)
        td_idx_a = td_idx_s
        td_idx_s_next = td_idx_s + 1
        td_idx_z_random = np.random.randint(0,i_0, batch_size)
        td_idx_episode_random = np.random.randint(0,i_1, batch_size)
        td_idx_s_random = np.random.randint(0,i_2, batch_size)
        #batch 
        z = torch.from_numpy(rb_z[td_idx_z, td_idx_episode]).to(self.device)
        s = torch.from_numpy(rb_states[td_idx_z, td_idx_episode, td_idx_s]).to(self.device)
        a = torch.from_numpy(rb_actions[td_idx_z, td_idx_episode, td_idx_a]).to(self.device)
        s_next = torch.from_numpy(rb_states[td_idx_z, td_idx_episode, td_idx_s_next]).to(self.device)
        s_e = torch.from_numpy(rb_states[td_idx_z_random, td_idx_episode_random, td_idx_s_random]).to(self.device)
      
        with torch.no_grad():
            a_next, _, _ = actor.get_action(s_next, z)
            # _, _, a_next_td = actor.get_action(s_next_td, z_td)
        loss= self.loss_td(s, a, s_next, a_next, s_e, z)
        # update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # moving average 
        self.exponential_moving_average()
        return 0.0, 0.0, 0.0

    def measure(self, s1, a1, s2, z):
        F1=self.C(s1,a1,s2,z)
        return F1
   
    # def measure_bij(self,s1,a1,s2,z):
    #     return self.C(s1,a1,s2,z)

