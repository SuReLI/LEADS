import torch
import torch.nn as nn
import numpy as np
import random
import torch.nn.functional as F
import torch.nn.init as init
from env.maze.maze import Maze
from env.mujoco.wmujoco import WMujoco
import math
import time
from utils.tools import make_gradient_clip_hook, gradient_clip_hook


def _initialize_weights( layer):
        # Calculer fan_avg
        fan_in, fan_out = init._calculate_fan_in_and_fan_out(layer.weight)
        fan_avg = (fan_in + fan_out) / 2.0
        
        # Utiliser la distribution uniforme avec le gain basé sur fan_avg
        gain = math.sqrt(1.0 / fan_avg)
        bound = math.sqrt(3.0) * gain
        # bound = 0.1
        init.uniform_(layer.weight, -bound, bound)
        
        # Initialiser les biais à 0 (optionnel)
        init.constant_(layer.bias, 0)

class MeasureNetwork(torch.nn.Module):
    def __init__(self, state_dim, action_dim, z_dim, feature_dim =16):
        super(MeasureNetwork, self).__init__()
        # self.mujoco = mujoco
        self.mujoco = False
        self.feature_dim = feature_dim
        # phi (s,a)
        self.phi_1=torch.nn.Linear(state_dim+action_dim+z_dim, 256)
        # init.normal_(self.phi_1.weight, mean=0, std=0.1)
        # _initialize_weights(self.phi_1)
        # torch.nn.init.kaiming_uniform_(self.phi_1.weight, a=1, mode='fan_in', nonlinearity='relu')

        self.phi_2=torch.nn.Linear(256, 64)
        # init.normal_(self.phi_2.weight, mean=0, std=0.1)
        # _initialize_weights(self.phi_2)
        # torch.nn.init.kaiming_uniform_(self.phi_2.weight, a=1, mode='fan_in', nonlinearity='relu')

        self.phi_3=torch.nn.Linear(64, feature_dim) 
        # init.normal_(self.phi_3.weight, mean=0, std=0.1)
        # _initialize_weights(self.phi_3)
        # torch.nn.init.kaiming_uniform_(self.phi_3.weight, a=1, mode='fan_in', nonlinearity='relu')


        # csi (s)
        self.csi_1 = torch.nn.Linear(state_dim, 256) 
        # init.normal_(self.csi_1.weight, mean=0, std=0.1)
        self.csi_2 = torch.nn.Linear(256, feature_dim)
        # init.normal_(self.csi_2.weight, mean=0, std=0.1)
        # self.csi_3 = torch.nn.Linear(64, feature_dim)
        # init.normal_(self.csi_3.weight, mean=0, std=0.1)

        # f dynamic csi 
        # self.inv_dynamic_csi = torch.nn.Linear(2*feature_dim, action_dim)
        # # f dynamic phi 
        # self.dynamic_phi = torch.nn.Linear(feature_dim , feature_dim)
    
    def forward(self, s1, a1, s2, z):
        phi = self.f_phi(s1, a1, z)
        # phi = F.normalize(phi, p=2, dim=1)
        csi = self.f_csi(s2)
        # csi = F.normalize(csi, p=2, dim=1)
        x= torch.einsum('ik,ik->i', phi, csi).unsqueeze(-1)/10.0
        return x
    
    def f_csi(self, x): 
        x_csi = x 
        # csi = self.featurizer(x)
        x_csi=torch.nn.functional.relu(self.csi_1(x_csi))
        # x_csi=torch.nn.functional.relu(self.csi_2(x_csi))
        # csi=torch.nn.functional.tanh(self.csi_3(x_csi))
        csi=self.csi_2(x_csi)
        return csi 
    
    def featurizer(self,x):
        return x[:,-3:]
       
    def f_phi(self, s1, a1, z) : 
        x_phi=torch.cat([s1, a1, z], dim=1) 
        x_phi=torch.nn.functional.relu(self.phi_1(x_phi))
        x_phi=torch.nn.functional.relu(self.phi_2(x_phi))
        # phi=torch.nn.functional.tanh(self.phi_3(x_phi))
        phi=self.phi_3(x_phi)
        return phi


    # def d_phi(self, x): 
    #     return self.dynamic_phi(x)
    
    # def d_csi(self, s1,s2): 
    #     x = torch.cat([s1, s2], dim=1) 
    #     return self.inv_dynamic_csi(x)
    
class Clearning:
    def __init__(self, state_dim, action_dim, z_dim, learning_rate=0.001, device='cpu', gamma=0.99, _lambda = 0.2 ):

        # self.mujoco = True if isinstance(env_class(), WMujoco) else False
        self.C = MeasureNetwork(state_dim, action_dim, z_dim).to(device)
        self.optimizer = torch.optim.Adam(self.C.parameters(), lr=learning_rate)
        self.loss_fn = nn.BCELoss(reduction='none')
        self.loss_fn_u = nn.BCEWithLogitsLoss()
        self.mse_loss = torch.nn.MSELoss()
        self.gamma=gamma
        self.device = device
        self._lambda=_lambda
        self.tar= 1e-9
        self.hooks=[]
        

        # Monte-Carlo C-learning
    # def L1(self, s_current, a_current, s_future, z):
    #     obj_pos = (2-self.gamma)*torch.log(torch.sigmoid(self.C(s_current, a_current, s_future, z))+1e-11)
    #     loss = - obj_pos.mean() 
    #     return loss

    # # Temporal Difference C-learning
    # def L2(self, s_current, a_current, s_next, a_next, s_random, z):
    #     with torch.no_grad():
    #         w=torch.exp(self.C(s_next,a_next,s_random,z))
    #         w = torch.clamp(w, 0, 20)
    #     pos = self.gamma*w*torch.log(torch.sigmoid(self.C(s_current,a_current,s_random, z))+1e-11)
    #     neg = 2*torch.log(1-torch.sigmoid(self.C(s_current,a_current,s_random, z))+1e-11)
    #     obj = pos + neg
    #     return -1.0 * torch.mean(obj)
    
    # def L_MC(self,s_current, a_current, s_future, z, batch_size) : 
    #     phi = self.C.f_phi(s_current, a_current, z)  # (batch_dim, repr_dim)
    #     csi =self.C.f_csi(s_future)  # (batch_dim, repr_dim)
    #     logits = torch.einsum('ik,jk->ij', phi, csi)  # <sa_repr[i], g_repr[j]> for all i,j
    #     prob = torch.sigmoid(logits)
    #     labels = torch.eye(batch_size, device=self.device)  # Ensure the tensor is on the same device as the input states
    #     loss = -(labels*torch.log(prob+1e-11)+ (1-labels)*torch.log(1-prob+1e-11))
    #     # loss = torch.sum(loss, dim = 1)
    #     return loss.mean()
         # Monte-Carlo C-learning
    def loss_mc(self, s_current, a_current, s_future, s_random, z):
        obj_pos = torch.log(torch.sigmoid(self.C(s_current, a_current, s_future, z))+self.tar).mean()
        obj_neg = torch.log(1-torch.sigmoid(self.C(s_current, a_current, s_random, z))+self.tar).mean()
        loss = - (obj_pos + obj_neg)
        return loss

    # Temporal Difference C-learning
    def loss_td(self, s_current, a_current, s_next, a_next, s_random, z):
        obj_a = (1 - self.gamma) * torch.log(torch.sigmoid(self.C(s_current, a_current, s_next, z))+self.tar)
        obj_b = torch.log(1 - torch.sigmoid(self.C(s_current, a_current, s_random, z))+self.tar)
        w = torch.exp(self.C(s_next,a_next,s_random,z))
        w = w.detach()  
        w = torch.clamp(w, 0, 50)
        obj_c = self.gamma * w * torch.log(torch.sigmoid(self.C(s_current, a_current, s_random, z))+self.tar)
        objective = obj_a + obj_b + obj_c
        return -1.0 * torch.mean(objective)
    
    # def model_based_loss(self, s_current, a_current, s_next,  a_next, z): 
    #     # phi 
    #     phi = self.C.f_phi(s_current, a_current, z)
    #     phi_next = self.C.f_phi(s_next, a_next, z)
    #     phi_next_hat = self.C.d_phi(phi)
    #     l_phi = self.mse_loss(phi_next, phi_next_hat)
    #     # csi 
    #     # csi = self.C.f_csi(s_current)
    #     # csi_next = self.C.f_csi(s_next)
    #     # a_current_hat = self.C.d_csi(csi, csi_next)
    #     # l_csi = self.mse_loss(a_current, a_current_hat)
    #     return l_phi 

    def update_vector(self,rb_states,rb_actions,rb_z, codebook, actor, batch_size, n_episode, lambda_c):
        i_0 = rb_states.shape[0]
        i_1 = rb_states.shape[1]
        i_2 = rb_states.shape[2]
        i_3 = rb_states.shape[3]
        # mc
        mc_idx_z = np.random.randint(0,i_0, batch_size)
        mc_idx_episode = np.random.randint(i_1-n_episode,i_1, batch_size)
        mc_idx_s = np.random.randint(0,i_2-1, batch_size)
        mc_idx_next_s = mc_idx_s + 1 
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
        s_mc = torch.from_numpy(rb_states[mc_idx_z, mc_idx_episode, mc_idx_s]).to(self.device)
        s_next_mc = torch.from_numpy(rb_states[mc_idx_z, mc_idx_episode, mc_idx_next_s]).to(self.device)
        a_mc = torch.from_numpy(rb_actions[mc_idx_z, mc_idx_episode, mc_idx_a]).to(self.device)
        s_future_mc = torch.from_numpy(rb_states[mc_idx_z, mc_idx_episode, mc_idx_future]).to(self.device)
        s_random_mc = torch.from_numpy(rb_states[mc_idx_z_random, mc_idx_episode_random, mc_idx_s_random]).to(self.device)
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
        z_td = torch.from_numpy(rb_z[td_idx_z, td_idx_episode]).to(self.device)
        s_td = torch.from_numpy(rb_states[td_idx_z, td_idx_episode, td_idx_s]).to(self.device)
        a_td = torch.from_numpy(rb_actions[td_idx_z, td_idx_episode, td_idx_a]).to(self.device)
        s_next_td = torch.from_numpy(rb_states[td_idx_z, td_idx_episode, td_idx_s_next]).to(self.device)
        s_random_td = torch.from_numpy(rb_states[td_idx_z_random, td_idx_episode_random, td_idx_s_random]).to(self.device)
        # z_td = z_mc
        # s_td = s_mc 
        # a_td = a_mc 
        # s_next_td = s_next_mc
        # s_random_td = s_random_mc
        with torch.no_grad():
            a_next_td, _, _ = actor.get_action(s_next_td, z_td)
        # # mc loss
        # loss_mc = self.L1(s_mc, a_mc, s_future_mc, z_mc)
        # # td loss
        # loss_td = self.L2(s_td, a_td, s_next_td, a_next_td, s_random_td, z_td)
        # # total loss
        # loss = lambda_c * loss_mc + (1-lambda_c) * loss_td
        # mc loss
        loss_mc = self.loss_mc(s_mc, a_mc, s_future_mc, s_random_mc, z_mc)
        # model based loss 
        # loss_mb = self.model_based_loss(s_td, a_td, s_next_td, a_next_td, z_td)
        # td loss
        loss_td = self.loss_td(s_td, a_td, s_next_td, a_next_td, s_random_td, z_td)
        # total loss
        loss = lambda_c*loss_mc + (1-lambda_c)*loss_td 
        # loss = self.L_MC(s_mc, a_mc, s_future_mc, z_mc,  batch_size)
        # update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss_mc.item(), loss_td.item()
        # return  0.0, 0.0, 0.0


    def measure(self, s1, a1, s2, z):
        return torch.exp(self.C(s1,a1,s2,z))
        # return self.C(s1,a1,s2,z)
    
        
   
    # def measure_bij(self,s1,a1,s2,z):
    #     return self.C(s1,a1,s2,z)

