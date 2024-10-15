import sys 
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.wandb_server import WandbServer
from utils.algorithm import Algorithm
from utils.sac_normal import SoftQNetwork, Actor
from utils.rnd import RND
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

class RND_A(Algorithm): 
    def __init__(self, algo, env_class, env_params, N, gamma, lr_theta, lr_q, lr_starts, tau, policy_frequency, rnd_frequency, target_frequency, noise, alpha, autotune, n_episode, batch_size, device, n_env, wandb_project_name, wandb_entity, seed):
        super().__init__(algo, env_class, env_params,  N, 1, None, gamma, lr_theta, n_episode,  batch_size, device, n_env, wandb_project_name, wandb_entity, seed)
        # sac 
        self.actor = Actor(self.env.observation_space.shape[0], self.env.action_space.shape[0], self.env.action_space).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_theta)
        self.qf1 = SoftQNetwork(self.env.observation_space.shape[0], self.env.action_space.shape[0]).to(device)
        self.qf2 = SoftQNetwork(self.env.observation_space.shape[0], self.env.action_space.shape[0]).to(device)
        self.qf1_target = SoftQNetwork(self.env.observation_space.shape[0], self.env.action_space.shape[0]).to(device)
        self.qf2_target = SoftQNetwork(self.env.observation_space.shape[0], self.env.action_space.shape[0]).to(device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        self.q_optimizer = torch.optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=lr_q)
        self.learning_starts = lr_starts
        self.tau = tau
        self.policy_frequency = policy_frequency 
        self.rnd_frequency = rnd_frequency
        self.target_frequency = target_frequency 
        self.noise = noise 
        self.alpha = alpha 
        self.autotune = autotune
        self.global_step = 0
        if self.autotune:
            self.target_entropy = -torch.prod(torch.Tensor(self.envs.single_action_space.shape).to(device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = torch.optim.Adam([self.log_alpha], lr=5e-4)
        # replay_buffer 
        self.replay_buffer_states = [[[]] for i in range(self.n_env)]
        self.replay_buffer_actions = [[[]] for i in range(self.n_env)]
        self.replay_buffer_rewards = [[[]] for i in range(self.n_env)]
        self.replay_buffer_dones = [[[]] for i in range(self.n_env)]
        # ngu module 
        self.rnd_module = RND(self.env.observation_space.shape[0])
        self.rnd_optimizer =torch.optim.Adam(self.rnd_module.parameters(), lr=lr_theta)
        
    def train(self):
        """ Sample in the environment """
        for epoch in range(self.N):
            # each env corresponds to a skill
            for episode in range(self.n_episode):
                    s , i= self.envs.reset()
                    for idx in range(self.n_env) : self.replay_buffer_states[idx][-1].append(s[idx])
                    for t in range(self.env.max_steps):
                        with torch.no_grad():
                            a, _, mean= self.actor.get_action(torch.tensor(s, dtype=torch.float32, device=self.device))
                        a = a.detach().cpu().numpy()
                        s,r,d,t,i= self.envs.step(a)
                        with torch.no_grad():
                            for idx in range(self.n_env) : 
                                ri= self.rnd_module.uncertainty_measure(torch.tensor(s[idx], dtype=torch.float32, device=self.device))
                                r[idx] = ri 
                        for idx in range(self.n_env) : self.replay_buffer_actions[idx][-1].append(a[idx]) #a
                        for idx in range(self.n_env) : self.replay_buffer_rewards[idx][-1].append(r[idx]) #r
                        for idx in range(self.n_env) : self.replay_buffer_dones[idx][-1].append(int(d[idx])) #d
                        for z_idx in range(self.n_skill) : self.replay_buffer_infos[z_idx][-1].append(i['pos'][z_idx]) #i
                        for idx in range(self.n_env) : 
                            self.replay_buffer_states[idx][-1].append(s[idx]) #s
                            self.update_coverage(s, z_idx)
                        # update actor 
                        self.update_sac(self.global_step)
                        # update ngu
                        rnd_loss = self.update_rnd(self.global_step)
                        # global step 
                        self.global_step+=1
                        self.sample_nb += self.n_env
                    # compute reward 
                    # with torch.no_grad():
                    #     for idx in range(self.n_env) : self.replay_buffer_rewards[idx][-1]=list(self.ngu_module.uncertainty_measure(torch.tensor(self.replay_buffer_states[idx][-1],dtype=torch.float32, device=self.device)).cpu().numpy().flatten())
                    # add new list for next sampling
                    for idx in range(self.n_env) : self.replay_buffer_actions[idx].append([])
                    for idx in range(self.n_env) : self.replay_buffer_rewards[idx].append([])
                    for idx in range(self.n_env) : self.replay_buffer_dones[idx].append([])
                    for idx in range(self.n_env) : self.replay_buffer_states[idx].append([])
                    for z_idx in range(self.n_skill) : self.replay_buffer_infos[z_idx].append([])
                    print('coverage : ', self.coverage(), 'sample : ', self.sample_nb, 'rnd_loss : ', rnd_loss)
                   
            # self.plot_maze(epoch)
            # wandb 
            wandb.log({"epoch": epoch, 
                        "coverage":self.coverage(),
                        "sample" : self.sample_nb})
                    

    def sample(self,batch_size):
        """ Sample a batch of transitions from the replay buffer """
        idx_env = np.random.choice(np.arange(self.n_env), size=(batch_size, 1))
        # sample
        s,a,r,d,s_next = [], [], [], [], []
        for idx in idx_env[:, 0]:
            # s1
            idx_episode = np.random.randint(0,len(self.replay_buffer_states[idx]))
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
        return(s, a, r, d, s_next)

    def update_sac(self, global_step): 
        if global_step > self.learning_starts:
            (s, a, r, d, s_next) = self.sample(self.batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = self.actor.get_action(s_next)
                qf1_next_target = self.qf1_target(s_next, next_state_actions)
                qf2_next_target = self.qf2_target(s_next, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
                next_q_value = r + (torch.ones_like(d) - d) * self.gamma * (min_qf_next_target).view(-1)
            qf1_a_values = self.qf1(s, a).view(-1)
            qf2_a_values = self.qf2(s,a).view(-1)
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
                    pi, log_pi, _ = self.actor.get_action(s)
                    qf1_pi = self.qf1(s, pi)
                    qf2_pi = self.qf2(s, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi).view(-1)
                    actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()
                    if self.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = self.actor.get_action(s)
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

    def update_rnd(self, global_step): 
        if global_step > self.learning_starts and global_step % self.rnd_frequency == 0:
            (s, a, r, d, s_next) = self.sample(self.batch_size)
            loss = self.rnd_module.loss(s)
            self.rnd_optimizer.zero_grad()
            loss.backward()
            self.rnd_optimizer.step()
            return loss.item()
        else : 
            return 0.0
        
    def plot_maze(self, epoch):
        # set np array 
        self.rb_list_to_np()
        # plot
        cmap = plt.cm.get_cmap('viridis')
        fig, ax1 = plt.subplots(1, 1 ,figsize=(10, 10))       
        s0 = self.envs.envs[0].reset()[0]
        rb_shape = self.replay_buffer_states_np.shape
        s = self.replay_buffer_states_np.reshape(rb_shape[0] *rb_shape[1] * rb_shape[2], rb_shape[3])
        with torch.no_grad():
            uncertainty = list(self.rnd_module.uncertainty_measure(torch.tensor(s, dtype=torch.float32).to(self.device)).cpu().numpy())
        # ax1
        sc = ax1.scatter(s[:, 0], s[ :, 1],c=uncertainty, cmap=cmap )
        colorbar = fig.colorbar(sc, ax=ax1, label='Exploration_bonus')

        for wall in self.envs.envs[0].walls:
                x1, y1, x2, y2 = wall
                ax1.plot([x1, x2], [y1, y2], color='black')
        ax1.set_xlim([-self.env.max_x, self.env.max_x])
        ax1.set_ylim([-self.env.max_y, self.env.max_y])
        # save figure
        fig.savefig(self.path+'epoch_{}.png'.format(epoch))
        # close fig 
        plt.close('all')

            
    

if __name__=='__main__': 
    from env.maze.maze import Maze
    env_class= Maze
    env_params = {'name': 'Ur'}
    N = 100
    sigma = 0.1
    n_skill = 6
    z_dim = n_skill
    gamma = 0.99
    lr_theta = 5e-4
    lr_q = 1e-3
    lr_starts = 5e2
    tau = 0.005
    policy_frequency = 2
    target_frequency = 1
    noise = 0.5
    alpha = 0.1
    autotune = False
    lr_a_entropy=5e-4
    n_sgd_clearning = 256
    n_sgd_discriminable_loss = 16
    n_sgd_entropy_loss = 16
    batch_size = 256
    n_episode = 5
    seed = 0
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    n_env = 5
    wandb_project_name = 'LDS'
    algo = 'rnd'
    wandb_entity = 'rnd'
    algo = RND_A(algo, env_class, env_params, N, n_skill, z_dim, gamma, lr_theta, lr_q, lr_starts, tau, policy_frequency, target_frequency, noise, alpha, autotune, n_episode, batch_size, device, n_env, wandb_project_name, wandb_entity, seed)
    algo.train()


