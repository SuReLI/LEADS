import sys 
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import env
from env.maze.maze import Maze
from utils.wandb_server import WandbServer
from utils.algorithm import Algorithm
from utils.sac import SoftQNetwork, Actor
from utils.codebook import Codebook
from discriminator import Discriminator
import math
import torch.nn.functional as F
import gym
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import wandb
import random
# import deepcopy
import copy

class DIAYN(Algorithm): 
    def __init__(self, algo, env_class, env_params, N, n_skill, z_dim, gamma, lr_theta, lr_q, lr_starts, tau, policy_frequency, target_frequency, noise, alpha, autotune, n_episode, batch_size, device, n_env, wandb_project_name, wandb_entity, seed, featurize):
        super().__init__(algo, env_class, env_params,  N, n_skill, z_dim, gamma, lr_theta, n_episode,  batch_size, device, n_env, wandb_project_name, wandb_entity, seed, featurize)
        # ensure z==n
        # discriminator
        self.discriminator = Discriminator(self.env.observation_space.shape[0],self.n_skill, self.env.name, featurize).to(self.device)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr_theta)
        self.loss_discriminator = torch.nn.CrossEntropyLoss()
        # sac 
        # codebook 
        self.codebook = Codebook(n=n_skill, d=z_dim, device=device, algo=algo)
        # actor
        self.actor = Actor(self.env.observation_space.shape[0], self.env.action_space.shape[0], self.z_dim,  self.env.action_space).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_theta)
        self.qf1 = SoftQNetwork(self.env.observation_space.shape[0], self.env.action_space.shape[0], self.z_dim).to(device)
        self.qf2 = SoftQNetwork(self.env.observation_space.shape[0], self.env.action_space.shape[0], self.z_dim).to(device)
        self.qf1_target = SoftQNetwork(self.env.observation_space.shape[0], self.env.action_space.shape[0], self.z_dim).to(device)
        self.qf2_target = SoftQNetwork(self.env.observation_space.shape[0], self.env.action_space.shape[0], self.z_dim).to(device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        self.q_optimizer = torch.optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=lr_q)
        self.learning_starts = lr_starts
        self.tau = tau
        self.policy_frequency = policy_frequency 
        self.target_frequency = target_frequency 
        self.noise = noise 
        self.alpha = alpha 
        self.autotune = autotune
        self.global_step = 0
        if self.autotune:
            self.target_entropy = -torch.prod(torch.Tensor(self.envs.single_action_space.shape).to(device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = torch.optim.Adam([self.log_alpha], lr=lr_theta)

        
    def train(self):
        """ Sample in the environment """
        # each env corresponds to a skill
        for epoch in range(self.N):
            for episode in range(self.n_episode):
                    z_omega=self.codebook.codebook_vectors.detach()
                    s , i= self.envs.reset()
                    for z_idx in range(self.n_skill) : self.replay_buffer_states[z_idx][-1].append(s[z_idx])
                    for z_idx in range(self.n_skill) : self.replay_buffer_z[z_idx].append(z_omega[z_idx].cpu().numpy())
                    for t in range(self.env.max_steps):
                        with torch.no_grad():
                            a, _, mean= self.actor.get_action(torch.tensor(s, dtype=torch.float32, device=self.device), z_omega)
                        a = a.detach().cpu().numpy()
                        s,r,d,t,i= self.envs.step(a)
                        with torch.no_grad():
                            r+=(torch.log(torch.sum(self.discriminator(torch.tensor(s, dtype=torch.float32, device=self.device))*z_omega, dim = 1)) - np.log(1/self.n_skill)).cpu().numpy()
                        for z_idx in range(self.n_skill) : self.replay_buffer_actions[z_idx][-1].append(a[z_idx]) #a
                        for z_idx in range(self.n_skill) : self.replay_buffer_rewards[z_idx][-1].append(r[z_idx]) #r
                        for z_idx in range(self.n_skill) : self.replay_buffer_dones[z_idx][-1].append(int(d[z_idx])) #d
                        for z_idx in range(self.n_skill) : self.replay_buffer_infos[z_idx][-1].append(i['pos'][z_idx]) #i
                        for z_idx in range(self.n_skill) : 
                            self.replay_buffer_states[z_idx][-1].append(s[z_idx]) #s
                            self.update_coverage(s, z_idx)
                        # update actor 
                        self.update_sac(self.global_step)
                        # update disciminator 
                        self.update_discriminator(self.global_step)
                        # global step 
                        self.global_step+=1
                        self.sample_nb += self.n_env
                    # add new list for next sampling
                    for z_idx in range(self.n_skill) : self.replay_buffer_actions[z_idx].append([])
                    for z_idx in range(self.n_skill) : self.replay_buffer_rewards[z_idx].append([])
                    for z_idx in range(self.n_skill) : self.replay_buffer_dones[z_idx].append([])
                    for z_idx in range(self.n_skill) : self.replay_buffer_states[z_idx].append([])
                    for z_idx in range(self.n_skill) : self.replay_buffer_infos[z_idx].append([])
            # plot 
            self.plot_maze(epoch) if self.maze_type else self.plot_mujoco(epoch)

            self.save_model(epoch) if (epoch % 5 ==0) else None
            
            # wandb 
            wandb.log({"epoch": epoch, 
                        "coverage":self.coverage(),
                        "sample" : self.sample_nb})
            # log 
            print("epoch: ", epoch, "coverage: ", self.coverage(), "sample: ", self.sample_nb)


    def sample(self,batch_size):
        """ Sample a batch of transitions from the replay buffer """
        # sample skills 
        idx_z = np.random.choice(np.arange(self.codebook.n), size=(batch_size, 1))
        z = self.codebook.codebook_vectors[idx_z[:,0]].detach()
        s,a,r,d,s_next = [], [], [], [], []
        for idx in idx_z[:, 0]:
            # s1
            idx_episode = np.random.randint(0,len(self.replay_buffer_z[idx]))
            idx_time_step= np.random.randint(0,len(self.replay_buffer_states[idx][idx_episode])-1)
            # add batch 
            s.append(torch.tensor(self.replay_buffer_states[idx][idx_episode][idx_time_step], dtype=torch.float32))
            a.append(torch.tensor(self.replay_buffer_actions[idx][idx_episode][idx_time_step], dtype=torch.float32))
            d.append(torch.tensor(self.replay_buffer_dones[idx][idx_episode][idx_time_step], dtype=torch.float32))
            r.append(torch.tensor(self.replay_buffer_rewards[idx][idx_episode][idx_time_step], dtype=torch.float32))
            s_next.append(torch.tensor(self.replay_buffer_states[idx][idx_episode][idx_time_step+1], dtype=torch.float32))
        s = torch.stack(s,dim=0).to(self.device)
        a = torch.stack(a,dim=0).to(self.device)
        d = torch.stack(d,dim=0).to(self.device)
        r = torch.stack(r,dim=0).to(self.device)
        s_next = torch.stack(s_next,dim=0).to(self.device)
        return(s, a, r, d, s_next, z)

    def update_sac(self, global_step): 
        if global_step > self.learning_starts:
            (s, a, r, d, s_next, z) = self.sample(self.batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = self.actor.get_action(s_next, z)
                qf1_next_target = self.qf1_target(s_next, next_state_actions, z)
                qf2_next_target = self.qf2_target(s_next, next_state_actions, z)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
                next_q_value = r + (torch.ones_like(d) - d) * self.gamma * (min_qf_next_target).view(-1)
            qf1_a_values = self.qf1(s, a, z).view(-1)
            qf2_a_values = self.qf2(s,a,z).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss
            self.q_optimizer.zero_grad()
            qf_loss.backward()
            self.q_optimizer.step()

            if global_step % self.policy_frequency == 0:  
                for _ in range(
                    self.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = self.actor.get_action(s,z)
                    qf1_pi = self.qf1(s, pi,z)
                    qf2_pi = self.qf2(s, pi,z)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi).view(-1)
                    actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()
                    if self.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = self.actor.get_action(s,z)
                        alpha_loss = (-self.log_alpha * (log_pi + self.target_entropy)).mean()
                        self.actor_optimizer.zero_grad()
                        alpha_loss.backward()
                        self.actor_optimizer.step()
                        alpha = self.log_alpha.exp().item()
            # update the target networks
            if global_step % self.target_frequency == 0:
                for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def update_discriminator(self, epoch): 
        (s, a, r, d, s_next, z) = self.sample(self.batch_size)
        z_pred = self.discriminator(s)
        loss = self.loss_discriminator(z_pred,z.argmax(dim=1))
        self.discriminator_optimizer.zero_grad()
        loss.backward()
        self.discriminator_optimizer.step()

            
    




