import torch
import torch.nn as nn
import numpy as np
from utils.rnd import RND


class NGU(torch.nn.Module):
    def __init__(self,  state_dim, action_dim, d_embedding=8, k = 10):
        super(NGU, self).__init__()
        self.rnd = RND(state_dim)
        self.L = 5 
        self.c = 0.001
        # embedding network
        self.l1 = nn.Linear(state_dim, 128)
        self.l2 = nn.Linear(128, d_embedding)
        # prediction network
        self.h1 = nn.Linear(2*d_embedding, 32)
        self.h2 = nn.Linear(32, action_dim)
        self.k = k
        self.epsilon = 1e-3

    
    def embedding(self, s):
        x = torch.relu(self.l1(s))
        # x = torch.relu(self.l2(x))
        x = self.l2(x)
        return x
    
    def h(self, s0, s1):
        x = torch.cat((s0, s1), 1)
        x = torch.relu(self.h1(x))
        x = self.h2(x)
        return x
       
    def r_i(self, s, s_episode, s_dm_1):
        if s_episode.shape[0] > self.k : 
            with torch.no_grad():
                alpha = self.rnd.uncertainty_measure(s)
                s = s.repeat(s_episode.shape[0],1)
                dists = self.distance_matrix_epoch(s, s_episode).unsqueeze(1)
                knn, s_dm = self.sum_k_nearest_epoch(dists, self.k, s_dm_1)
            r_episodic  = 1/(torch.sqrt(knn) + self.c)
            r = r_episodic * torch.min(torch.max(alpha,torch.ones_like(alpha)),torch.ones_like(alpha)*self.L)
            return r.item(), s_dm
        else : 
            return 0.0, 0.0
    
    def sum_k_nearest_epoch(self, dist, k, s_dm_1):
        k_nearest_neighbors, _ = torch.topk(dist, k=k, dim=0, largest=False)
        k = torch.sum(k_nearest_neighbors, dim = 0)
        k_running = s_dm_1/dist.shape[0]
        s_dm = s_dm_1 + k
        return self.epsilon/(k**2/k_running**2 + self.epsilon), s_dm
    
    def distance_matrix_epoch(self, s, s_epoch):
        x = self.embedding(s)
        x_epoch = self.embedding(s_epoch)
        dist = torch.norm(x - x_epoch, dim=1)
        return dist
    
    def uncertainty_measure(self, s):
        with torch.no_grad():
            alpha = self.rnd.uncertainty_measure(s)
            dist_matrix = self.distance_matrix(s)
            knn = self.sum_k_nearest(dist_matrix, k=self.k)
        r_episodic  = 1/(torch.sqrt(knn) + self.c)
        r = r_episodic * torch.min(torch.max(alpha,torch.ones_like(alpha)),torch.ones_like(alpha)*self.L)
        return r

    def loss(self,s,a,s_next): 
        rnd_loss = self.rnd.loss(s)
        # NGU loss 
        s0 = self.embedding(s)
        s1 = self.embedding(s_next)
        h_loss = (self.h(s0, s1) - a)**2 # continuous action space
        return rnd_loss + h_loss.mean()
        
    def distance_matrix(self, s):
        x = self.embedding(s)
        dist = torch.sum(x**2, 1).view(-1, 1) + torch.sum(x**2, 1).view(1, -1) - 2 * torch.mm(x, x.t())
        return torch.sqrt(dist)


    def sum_k_nearest(self, dist_matrix, k=1):
        _, indices = dist_matrix.sort(dim=1)
        k_nearest_indices = indices[:, 1:k+1] 
        k_nearest_values = torch.gather(dist_matrix, 1, k_nearest_indices)
        sum_k_nearest = k_nearest_values.sum(dim=1, keepdim=True)
        mean_k_nearest_2 = sum_k_nearest.mean()**2
        k_2 = sum_k_nearest**2
        sum_k = self.epsilon/(k_2/mean_k_nearest_2 + self.epsilon)
        return sum_k


