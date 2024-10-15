import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.maze.maze import Maze
from env.mujoco.wfetch import WFetch
from env.mujoco.whand import WHand
from env.mujoco.whopper2 import Whopper
# from env.mujoco.wfinger import WFinger
from env.mujoco.wfetch_push import WFetch_Push
from env.mujoco.whalfcheetah import Whalfcheetah
from utils.wrapper import CustomEnvWrapper, make_env
from utils.tools import aplatir
from utils.wandb_server import WandbServer
from utils.algorithm import Algorithm
from clearning import Clearning
from utils.create_colors import get_n_distinct_colors_deterministic
from sklearn.decomposition import PCA
import colorednoise as cn
# from rc_learning import Clearning
from codebook import Codebook
from sac import Actor
import math
import torch.nn.functional as F
import gym
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
import time
import wandb
import random
# import deepcopy
import copy




class LDS(Algorithm):
    def __init__(self, env_class, env_params, N, sigma, n_skill, z_dim, lambda_d, lambda_a, lambda_e, beta, lambda_e_archive, lambda_o, gamma, lambda_clearning, lr_theta, lr_skill, lr_c, n_episode, n_c_r, 
                n_update, n_sgd_clearning, n_sgd_discriminable_loss, n_sgd_entropy_loss, n_archive, batch_size_clearning, batch_size_loss, device, n_env, wandb_project_name, wandb_entity, seed = 0, featurize = False):
        super().__init__('lds', env_class, env_params,  N, n_skill, z_dim, gamma, lr_theta, n_episode,  batch_size_loss, device, n_env, wandb_project_name, wandb_entity, seed, featurize) 
        """ Initialize the LDS algorithm
        N: number of epochs
        sigma: standard deviation of the gaussian ball around the codebook vectors
        n_skill: number of skills
        z_dim: dimension of the skill space
        lambda_d: weight of the discriminable loss
        lambda_e: weight of the entropy loss (exploration loss)
        beta : our criteria of concentrated mass for one skill
        lambda_e_archive: entropy on archive
        lambda_o: weight of the orthonormality loss
        gamma: discount factor
        lambda_clearning: weight between mc loss and td loss
        lr_theta: learning rate of the actor
        lr_skill: learning rate of the codebook
        lr_c: learning rate of the successor state measure
        n_episode: number of episodes to sample between each update
        n_sgd: number of gradient descent steps for each update
        n_archive: number of states to store in the archive
        batch_size: batch size for the replay buffer
        device: device to use for the computations
        n_env: number of parallel environments"""
        # set_all_seeds
        self.N = N
        self.sigma = sigma
        self.n_skill = n_skill
        self.z_dim = z_dim
        self.lambda_d = lambda_d
        self.lambda_a = lambda_a
        self.lambda_e = lambda_e
        self.beta = beta
        self.lambda_e_archive = lambda_e_archive
        self.lambda_o = lambda_o
        self.gamma = gamma
        self.lambda_clearning = lambda_clearning
        self.lr_theta = lr_theta
        self.lr_skill = lr_skill
        self.lr_c = lr_c
        self.n_episode = n_episode
        self.n_c_r = n_c_r
        self.n_update = n_update
        self.n_sgd_clearning = n_sgd_clearning
        self.n_sgd_discriminable_loss = n_sgd_discriminable_loss
        self.n_sgd_entropy_loss = n_sgd_entropy_loss
        self.batch_size_clearning = batch_size_clearning
        self.batch_size_loss = batch_size_loss
        self.device = device
        self.n_env = n_skill
        # archive
        self.n_archive = n_archive
        self.archive = [[] for _ in range(self.n_skill)]
        self.info_archive = [[] for _ in range(self.n_skill)]
        self.list_archive = []
        # replay buffer np array
        self.replay_buffer_states_np = None
        self.replay_buffer_actions_np = None
        self.replay_buffer_z_np = None
        self.replay_buffer_s_u_np = None
        self.replay_buffer_a_u_np = None
        # old_codebook
        self.codebook = Codebook(n_skill, z_dim, device, std=sigma, learning_rate=lr_skill)
        self.list_codebook = []
        # successor state measure
        self.c_learning = Clearning(self.env.observation_space.shape[0], self.env.action_space.shape[0], z_dim, learning_rate=lr_c, device=device, gamma=gamma, _lambda=lambda_clearning, featurizer=featurize, env_name=self.env.name)
        self.list_c_learning = []
        # actor
        self.actor = Actor(self.env.observation_space.shape[0], self.env.action_space.shape[0], self.z_dim,  self.env.action_space).to(self.device)
        self.list_actor = []
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_theta)
        # sync envs
        self.s0 = torch.tensor(self.envs.envs[0].reset()[0], dtype=torch.float32, device=self.device).repeat(n_skill,1)
        # wandb server  
        self.wandb_hp = {'N': N, 'sigma': sigma, 'n_skill': n_skill, 'z_dim': z_dim, 'lambda_d': lambda_d, 'lambda_e': lambda_e, 'lambda_e_archive': lambda_e_archive, 'lambda_o': lambda_o, 'gamma': gamma, 'lambda_clearning': lambda_clearning, 'lr_theta': lr_theta, 'lr_skill': lr_skill, 'lr_c': lr_c, 'n_episode': n_episode, 'n_update': n_update, 'n_sgd_clearning': n_sgd_clearning, 'n_sgd_discriminable_loss': n_sgd_discriminable_loss, 'n_sgd_entropy_loss': n_sgd_entropy_loss, 'batch_size_loss': batch_size_loss, 'device': device, 'n_env': n_env}
        wandb.config = self.wandb_hp
        # self.wandb_server = WandbServer(wandb_project_name, wandb_entity, config=self.wandb_hp)
        # make dir 
        self.path_measure = self.path+'/measure/'
        self.path_uncertainty =self.path+'/uncertainty/'
        self.path_main = self.path+'/main/'
        self.path_reach = self.path+'/reach/'
        self.path_model = self.path+'/model/'

        os.makedirs(self.path, exist_ok=True)
        os.makedirs(self.path_measure, exist_ok=True)
        os.makedirs(self.path_uncertainty, exist_ok=True)
        os.makedirs(self.path_main, exist_ok=True)
        os.makedirs(self.path_reach, exist_ok=True)
        os.makedirs(self.path_model, exist_ok=True)

        
    def set_all_seeds(self, seed):
        random.seed(seed)          # Python
        np.random.seed(seed)       # Numpy
        torch.manual_seed(seed)    # PyTorch
        # Si vous utilisez CUDA pour PyTorch
        # if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def sample_env(self):
        """ Sample in the environment """
        # each env corresponds to a skill
        for episode in range(self.n_episode):
                eps_tm = np.concatenate([ np.concatenate([cn.powerlaw_psd_gaussian(1.0, self.env.max_steps +1 )[:, None] for _ in range(self.envs.single_action_space.shape[0])], axis=1)[None, :] for _ in range(self.n_env)], axis=0)
                z_omega=self.codebook.codebook_vectors.detach()
                z_tilde = self.codebook.sample_gaussian_around(n=self.n_skill,vec=z_omega).detach()
                s , i= self.envs.reset()
                for z_idx in range(self.n_skill) : self.replay_buffer_states[z_idx][-1].append(s[z_idx])
                for z_idx in range(self.n_skill) : self.replay_buffer_infos[z_idx][-1].append(i['pos'][z_idx])
                for z_idx in range(self.n_skill) : self.replay_buffer_z[z_idx].append(z_tilde[z_idx].cpu().numpy())
                for t in range(self.env.max_steps):
                    with torch.no_grad():
                        eps = eps_tm[np.arange(self.n_env), t]
                        a, _, mean= self.actor.get_action(torch.tensor(s, dtype=torch.float32, device=self.device), z_tilde, eps = torch.tensor(eps, dtype=torch.float32, device=self.device))
                        a = a.detach().cpu().numpy()
                        s,r,d,t,i= self.envs.step(a)
                        # print('i : ', i['pos'][0])
                        # input()
                        for z_idx in range(self.n_skill) : self.replay_buffer_actions[z_idx][-1].append(a[z_idx])
                        for z_idx in range(self.n_skill) : self.replay_buffer_infos[z_idx][-1].append(i['pos'][z_idx])
                        for z_idx in range(self.n_skill) : 
                            self.replay_buffer_states[z_idx][-1].append(s[z_idx])
                            self.update_coverage(s, z_idx)
                    self.sample_nb += self.n_env
                # add new list for next sampling
                for z_idx in range(self.n_skill) : self.replay_buffer_states[z_idx].append([])
                for z_idx in range(self.n_skill) : self.replay_buffer_actions[z_idx].append([])
                for z_idx in range(self.n_skill) : self.replay_buffer_infos[z_idx].append([])
      
            
    
    def train(self):
        """ Train the agent """
        for epoch in range(self.N):
            hs_lb_global = 0
            hsz_lb_global = 0
            orthonormal_codebook_loss_global = 0
            mc_global = 0
            td_global = 0
            if epoch!=0:
                # add last model to list 
                self.list_actor.append(copy.deepcopy(self.actor))
                self.list_codebook.append(copy.deepcopy(self.codebook))
                self.list_c_learning.append(copy.deepcopy(self.c_learning))
                self.list_archive.append(copy.deepcopy(self.archive))
                for u in range(self.n_update):
                    # #################################################### THETA LOSS#################################################### 
                    # -H(S|Z)
                    hsz_lb = self.H_SZ_loss()
                    # H(S_archive)
                    hs_archive = self.H_archive_loss() 
                    # H(A|S_archive)
                    h_a = self.entropy_policy_archive()
                    # H(A|S)
                    # has = self.H_AS_loss()
                    # #################################################### Z LOSS#################################################### 
                    # H(S)
                    hs_lb= self.H_S_loss()
                    # orthnormal loss
                    orthonormal_codebook_loss = self.codebook.orthonormal_loss()
                    # loss codebook
                    loss_codebook =  self.lambda_o*orthonormal_codebook_loss + hs_lb
                    # loss actor 
                    loss_actor =  hs_archive + self.lambda_e_archive*h_a + self.lambda_d*hsz_lb 
                    # #################################################### Z UPDATE#################################################### 
                    # zero codebook 
                    self.codebook.optimizer.zero_grad()
                    # backward codebook
                    loss_codebook.backward()
                    # update codebook
                    self.codebook.optimizer.step()
                    # #################################################### THETA UPDATE####################################################
                    # zero actor optimizer
                    self.actor_optimizer.zero_grad()
                    # # backward actor
                    loss_actor.backward()
                    # update actor
                    self.actor_optimizer.step()
                    # metrics 
                    hs_lb_global += hs_lb.item()
                    hsz_lb_global += hsz_lb.item()
                    orthonormal_codebook_loss_global += orthonormal_codebook_loss.item()
                    # delete hook 
                    for hook in self.actor.hooks : hook.remove()
                    self.actor.hooks = []
                    for hook in self.c_learning.hooks : hook.remove()
                    self.c_learning.hooks = []
                    # re-update c-learning 
                    if epoch >= 0 : 
                        for _ in range(self.n_c_r) : 
                            mc_loss, td_loss = self.update_measure(0.0)
                    print('update : ', u)

            # sample env 
            self.sample_env()
            # set np array 
            self.rb_list_to_np(epoch)
            # c-learning update 
            for i in range(self.n_sgd_clearning):
                mc_loss, td_loss = self.update_measure(self.lambda_clearning) 
                mc_global += mc_loss
                td_global += td_loss
            # build archive 
            self.build_archive(epoch) if (epoch % 1 ==0) else None 
            # plot
            self.plot(epoch) 
            # log
            print("epoch: ", epoch, "mc_global: ", mc_global, "td_global: ", td_global, "hs_lb: ", hs_lb_global, "hsz_lb: ", 
                    hsz_lb_global, "orthonormal_codebook_loss: ", orthonormal_codebook_loss_global, 'coverage : ', 
                    self.coverage(), 'sample : ', self.sample_nb)
            # wandb 
            wandb.log({"epoch": epoch, 
                       "mc_global": mc_global, 
                       "td_global": td_global, 
                       "hs_lb" : hs_lb_global, 
                       "hsz_lb" : hsz_lb_global, 
                       "orthonormal_codebook_loss" : orthonormal_codebook_loss_global,
                       "coverage":self.coverage(),
                       "sample" : self.sample_nb})
            # remove data 
            # self.remove_data(epoch)
            # save model
            self.save_model(epoch) if (epoch % 5 ==0) else None

    def update_measure(self, lambda_c):
        mc_loss, td_loss = self.c_learning.update_vector(self.replay_buffer_states_np,self.replay_buffer_actions_np,self.replay_buffer_z_np, self.codebook, self.actor,self.batch_size_clearning, self.n_episode, lambda_c)
        return(mc_loss, td_loss)
        

    def measure(self, s1, a1, s2, z):
        return(self.c_learning.measure(s1, a1, s2, z))
    
    def measure_fix(self, s1, s2, z, m = False):
        a1, log_prob1, mean1 = self.actor.get_action(s1, z)
        # measure
        measure = self.c_learning.measure(s1, a1, s2, z) if not m else self.c_learning.measure(s1, mean1, s2, z)
        return(measure)    
    
    
    def measure_max(self, s1, s2, z):
        a1, log_prob1, mean1 = self.actor.get_action(s1, z)
        a2, log_prob2, mean2 = self.actor.get_action(s2, z)
        # measure
        measure1 = self.c_learning.measure(s1, a1, s2, z)
        measure2 = self.c_learning.measure(s2, a2, s1, z)
        return(torch.max(measure1, measure2))

    def measure_codebook(self, s1, s2):
        """ Return the measure measure_max for each skill in the codebook """
        # get codebook vectors
        codebook_vectors = self.codebook.codebook_vectors
        # measure for each skill
        measure = [self.measure_max(s1, s2, codebook_vectors[i].repeat(s1.shape[0],1)) for i in range(self.codebook.n)]
        # concatenate
        measure = torch.concat(measure, axis=1)
        return(measure)
    
    def measure_codebook_fix(self, s1, s2, m = False):
        """ Return the measure measure_max for each skill in the codebook """
        # get codebook vectors
        codebook_vectors = self.codebook.codebook_vectors
        # measure for each skill
        measure = [self.measure_fix(s1, s2, codebook_vectors[i].repeat(s1.shape[0],1), m) for i in range(self.codebook.n)]
        # concatenate
        measure = torch.concat(measure, axis=1)
        return(measure)
 
    def H_S_loss(self):
        """ Return the entropy loss """
        # sample s 
        _, s , _ , idx_z= self.sample(self.batch_size_loss, rho_stationary = False) # sample from the full dataset
        # full vector of skills
        z_full_codebook = [self.codebook.codebook_vectors[i].repeat(self.batch_size_loss,1) for i in range(self.codebook.n)]
        # sample s_m from each skill 
        s_m = [self.sample_from_z(i, self.batch_size_loss) for i in range(self.codebook.n)]
        # compute measure
        measure_full_codebook = [self.measure_max(s_m[i], s ,z_full_codebook[i]) for i in range(self.codebook.n)]
        # concatenate
        measure_full_codebook = torch.cat(measure_full_codebook, axis=1)
        # normalize 
        measure_full_codebook = (measure_full_codebook - torch.min(measure_full_codebook,dim =0).values.detach())/(torch.max(measure_full_codebook,dim =0).values.detach()- torch.min(measure_full_codebook,dim =0).values.detach())
        # bijection 
        # clamp 
        # measure_full_codebook = torch.clamp(measure_full_codebook,min=0.0, max=50.0)
        # exponential
        e_measure_full_codebook = torch.exp(measure_full_codebook)
        # e max
        e_max_full_codebook = torch.pow(torch.max(e_measure_full_codebook,axis=1).values,self.beta/(self.beta+1))
        # frac 
        frac = e_max_full_codebook / (torch.sum(e_measure_full_codebook, axis=1))
        # entropy
        lb_entropy = torch.log(frac+1e-9)
        # policy entropy 
        # logp_zi = [self.actor.get_action(s_m[i], self.codebook.codebook_vectors[i].repeat(self.batch_size_loss,1))[1].mean() for i in range(self.codebook.n)]
        # has = torch.stack(logp_zi).sum()
        # loss
        loss = - lb_entropy.mean() 
        return loss
    
    def sample_from_z(self, z_idx, batch_size):
        """ Sample a batch of transitions from the distribution z """
        # stationary data 
        states_z = self.replay_buffer_states[z_idx][len(self.replay_buffer_z[z_idx])-self.n_episode:len(self.replay_buffer_z[z_idx])]
        states_z = np.concatenate(states_z, axis=0)
        # sample 
        idx = np.random.randint(0, states_z.shape[0], size=(batch_size))
        s = states_z[idx]
        return(torch.tensor(s, dtype=torch.float32, device=self.device))

    
    def H_archive_loss(self):
        #  sample from archive 
        s, sa, z, idx_z= self.sample_archive(self.batch_size_loss, close = True)
        _, logp, _ = self.actor.get_action(s,z)
        # compute measure
        measure= self.measure_fix(s, sa, z)
        # measure = (measure - measure.min().detach())/(measure.max().detach()-measure.min().detach())
        # compute measure over full codebook
        measure_codebook = self.measure_codebook_fix(s, sa)
        # measure_codebook = (measure_codebook - torch.min(measure_codebook, dim = 0).values.detach())/(torch.max(measure_codebook, dim = 0).values.detach()- torch.min(measure_codebook, dim = 0).values.detach())
        sum_measure_codebook = torch.sum(measure_codebook, dim=1)
        # lower bound
        lb = torch.log(measure/ (1+sum_measure_codebook) + 1e-6)
        loss = - self.lambda_a*torch.mean(lb) + self.lambda_e*logp.mean()
        return(loss)

   
    
    def sample_archive(self,batch_size, close = False ):
        """ Sample a batch of transitions from the replay buffer """
        i_0 = self.replay_buffer_states_np.shape[0]
        i_1 = self.replay_buffer_states_np.shape[1]
        i_2 = self.replay_buffer_states_np.shape[2]
        i_3 = self.replay_buffer_states_np.shape[3]
        archive_np = np.array(self.archive)
        i_a_1 = archive_np.shape[1]
        idx_z = np.random.randint(0,i_0, batch_size)
        # idx_z_random = np.random.randint(0,i_0, batch_size)
        # idx_episode = np.random.randint(0,i_1, batch_size)
        idx_episode = np.random.randint(i_1-self.n_episode,i_1, batch_size)
        idx_s = np.random.randint(0,i_2-1, batch_size)
        idx_sa = np.random.randint(0,i_a_1, batch_size)
        # tensor 
        z = self.codebook.codebook_vectors[idx_z]
        s = torch.from_numpy(self.replay_buffer_states_np[idx_z, idx_episode, idx_s].astype(np.float32)).to(self.device)
        sa = torch.from_numpy(archive_np[idx_z, idx_sa].astype(np.float32)).to(self.device)
        return(s, sa, z, idx_z)
    
    
    def entropy_policy_archive(self):
        #  sample from archive 
        s, sa, z, idx_z= self.sample_archive(self.batch_size_loss)
        # entropy 
        a1, log_prob1, mean1 = self.actor.get_action(sa, z)
        return log_prob1.mean()
    
    def H_SZ_loss(self):
        """ Return the discriminable loss """
        # sample 
        s1, s2, z , idx_z= self.sample(self.batch_size_loss, rho_stationary = True) # sample from the accurate distribution
        # policy entropy 
        # _, logp1, _ = self.actor.get_action(s1,z)
        # _, logp2, _ = self.actor.get_action(s2,z)
        # compute measure
        measure= self.measure_max(s1, s2, z )
        # measure = (measure - measure.min().detach())/(measure.max().detach()-measure.min().detach())
        # compute measure over full codebook
        measure_codebook = self.measure_codebook(s1, s2)
        # measure_codebook = (measure_codebook - torch.min(measure_codebook, dim = 0).values.detach())/(torch.max(measure_codebook, dim = 0).values.detach()- torch.min(measure_codebook, dim = 0).values.detach())
        sum_measure_codebook = torch.sum(measure_codebook, dim=1)
        # lower bound
        lb = torch.log(measure/ (1+sum_measure_codebook) + 1e-6)
        # compute loss
        loss = - torch.mean(lb) 
        return(loss)

    
    def H_AS_loss(self):
        s, _ , z , _= self.sample(self.batch_size_loss) 
        a1, log_prob1, mean1 = self.actor.get_action(s, z)
        return log_prob1.mean()


    def sample(self,batch_size, rho_stationary = False):
        """ Sample a batch of transitions from the replay buffer """
        i_0 = self.replay_buffer_states_np.shape[0]
        i_1 = self.replay_buffer_states_np.shape[1]
        i_2 = self.replay_buffer_states_np.shape[2]
        idx_z = np.random.randint(0,i_0, batch_size)
        idx_episode_1 = np.random.randint(i_1-self.n_episode,i_1, batch_size)
        idx_episode_2 = np.random.randint(i_1-self.n_episode,i_1, batch_size)
        idx_s1 = np.random.randint(0,i_2-1, batch_size)
        idx_s2 = np.random.randint(0,i_2-1, batch_size)
        z = self.codebook.codebook_vectors[idx_z]
        s1 = torch.from_numpy(self.replay_buffer_states_np[idx_z, idx_episode_1, idx_s1].astype(np.float32)).to(self.device)
        s2 = torch.from_numpy(self.replay_buffer_states_np[idx_z, idx_episode_2, idx_s2].astype(np.float32)).to(self.device)
        return(s1, s2, z, idx_z)    

    def rb_list_to_np(self,epoch):
        list_rb_states = copy.deepcopy(self.replay_buffer_states)
        list_rb_actions = copy.deepcopy(self.replay_buffer_actions)
        list_rb_s_u = copy.deepcopy(self.replay_buffer_states)
        list_rb_infos = copy.deepcopy(self.replay_buffer_infos)
        list_rb_z = copy.deepcopy(self.replay_buffer_z)
        for i in range(len(list_rb_states)): 
            list_rb_states[i]=list_rb_states[i][-self.n_episode:-1]
            list_rb_actions[i]=list_rb_actions[i][-self.n_episode:-1]
            list_rb_infos[i]=list_rb_infos[i][-self.n_episode:-1]
            list_rb_z[i]=list_rb_z[i][-self.n_episode:-1]
            list_rb_s_u[i]=list_rb_s_u[i][-self.n_episode:-1]
        self.replay_buffer_z_np = np.array(list_rb_z)
        self.replay_buffer_states_np = np.array(list_rb_states)
        self.replay_buffer_actions_np = np.array(list_rb_actions)
        self.replay_buffer_s_u_np = np.array(list_rb_s_u)
        self.replay_buffer_infos_np = np.array(list_rb_infos)
        



    def build_archive(self, epoch ,n_last = 50):
        """ Build the archive based using the last ones """
        for idx in range(self.n_skill): 
            total_episodes = len(self.replay_buffer_z[idx])
            # states = [torch.tensor(np.array(self.replay_buffer_states[idx][episode][-n_last:])).to(self.device) for episode in range(max(total_episodes-self.n_episode,0),total_episodes)]
            states = [torch.tensor(np.array(self.replay_buffer_states[idx][episode][-n_last:]), dtype = torch.float32).to(self.device) for episode in range(0,total_episodes)]
            infos = [np.array(self.replay_buffer_infos[idx][episode][-n_last:]) for episode in range(0,total_episodes)]
            # states = [torch.tensor(np.array(self.replay_buffer_states[idx][episode][:])).to(self.device) for episode in range(0,total_episodes)]
            states = torch.cat(states,  dim=0)
            infos = np.concatenate(infos, axis=0)
            uncertainty_measure = self.uncertainty_measure(states, idx, epoch )
            values, indices = torch.topk(uncertainty_measure.squeeze(1), 1)
            self.archive[idx]=states[indices].cpu().numpy()
            self.info_archive[idx]=np.expand_dims(infos[indices], axis=0)


    def uncertainty_measure(self, s, i , epoch ):
        if self.env.name == 'hopper' :
            k_w = 100.0
        elif self.env.name == 'half_cheetah' :
            k_w = 100.0 
        else : 
            k_w = 100.0
        with torch.no_grad():
            # Distribution from s0
            s0 = torch.tensor(self.envs.envs[0].reset()[0] , dtype=torch.float32, device=self.device).repeat(s.shape[0],1)
            z_i = self.codebook.codebook_vectors[i].repeat(s0.shape[0], 1).to(self.device).detach()
            m = self.measure_fix(s0,s,z_i, m = True) 
            m_n = (m-m.min())/(m.max()-m.min())
            measure_codebook = self.measure_codebook_fix(s0,s, m = True) 
            measure_codebook_n = (measure_codebook - torch.min(measure_codebook, dim = 0)[0])/(torch.max(measure_codebook, dim = 0)[0]- torch.min(measure_codebook, dim = 0)[0])
            m_n_r = m_n.repeat(1,self.n_skill)
            dl_l = self.dist(m_n_r,measure_codebook_n, k=k_w)
            kl = torch.sum(dl_l, dim = 1).unsqueeze(1)
            kl = (kl - kl.min())/(kl.max()-kl.min())
            if epoch!=0 :
                # Distribution around archive 
                s_a_i = [ torch.tensor(self.archive[idx][0], dtype=torch.float32, device=self.device).repeat(s.shape[0],1) for idx in range(self.n_skill) ]
                z_i_l= [self.codebook.codebook_vectors[idx].repeat(s.shape[0],1) for idx in range(self.n_skill)]
                measure_i = [ self.measure_fix(s_a_i[idx], s, z_i_l[idx], m = True) for idx in range(self.n_skill)]
                measure_i_n = [(measure_i[idx]-measure_i[idx].min())/(measure_i[idx].max()-measure_i[idx].min()) for idx in range(self.n_skill)]
                kl_i = [self.dist(measure_i_n[i],measure_i_n[idx], k= k_w) for idx in range(self.n_skill)]
                kl_i_n = [(kl_i[idx]-kl_i[idx].min())/(kl_i[idx].max()-kl_i[idx].min()) for idx in range(self.n_skill)]
                kl_cat = torch.cat(kl_i_n,dim=1)
                # kl_cat_nearest = torch.topk(kl_cat,k=self.n_skill-1, dim =1, largest = False)[0]
                kl_d = torch.sum(kl_cat, dim =1).unsqueeze(1)
                kl_d = (kl_d - kl_d.min())/(kl_d.max()-kl_d.min())
                # # m_old
                # s_a_i_old = torch.tensor(self.archive[i][0], dtype=torch.float32, device=self.device).repeat(s.shape[0],1)
                # m_from_a =  self.measure_fix(s_a_i_old, s, z_i, m = True) 
                # m_from_a_n = (m_from_a-m.min())/(m_from_a.max()-m_from_a.min())
                # Distribution induced by all the last policies
                h_m = self.measure_from_past(s0, s, i)
                h_kl = self.dist(measure_i_n[i], h_m, k=k_w) 
                # h_kl = self.dist(m_n, h_m, k=100.0) 
                h_kl  = (h_kl - h_kl.min())/(h_kl.max()-h_kl.min())
                if self.env.name == 'hopper' :
                    kl = (kl + 2*kl_d + 2*h_kl)/4
                elif self.env.name == 'finger' :
                    kl = (kl + 2*kl_d + 2*h_kl)/5
                elif self.env.name == 'half_cheetah' :
                    kl = (kl + 2*kl_d + 2*h_kl)/4
                elif self.env.name == 'hand' :
                    kl =  kl_d 
                else :
                    kl = (kl + 2*kl_d + 2*h_kl)/4 
                # kl = (kl + 2*kl_d + 4*h_kl)/7
                # kl=kl
            else : 
                m_from_a_n = 1
                h_kl = 1
                kl_d = 1
            return kl 
        
    def dist(self,m1,m2, k = 10.0):
        return torch.log((m1 + 1e-1)/(k*m2 + 1e-1 ))
        

    def measure_from_past(self, s1, s2, i):
        measure = 0
        for e in range(len(self.list_codebook)) :
        # e=-1
            z_i = self.list_codebook[e].codebook_vectors
            actor = self.list_actor[e]
            c_learning = self.list_c_learning[e]
            for idx in range(self.n_skill):
                a1, log_prob1, mean1 = self.actor.get_action(s1, z_i[idx].repeat(s1.shape[0],1))
                m = c_learning.measure(s1, mean1, s2, z_i[idx].repeat(s1.shape[0],1))
                measure += 0.99**(len(self.list_codebook)-e)*(m-m.min())/(m.max()-m.min())
                # measure += (m-m.min())/(m.max()-m.min())
        return (measure - measure.min())/(measure.max()-measure.min())
    
    def remove_data(self, epoch):
        """ Remove data from the replay buffer """
        for idx in range(self.n_skill):
            for e in range(self.n_episode):
                # remove 1/2 of the data in each episode
                self.replay_buffer_states[idx][e] = self.replay_buffer_states[idx][e][::8]
                self.replay_buffer_actions[idx][e] = self.replay_buffer_actions[idx][e][::8]
                self.replay_buffer_infos[idx][e] = self.replay_buffer_infos[idx][e][::8]
                self.replay_buffer_z[idx][e] = self.replay_buffer_z[idx][e][::8]
            

    def save_model(self, epoch):
        # save model
        torch.save(self.actor.state_dict(), self.path_model + 'actor_{}.pt'.format(epoch)) # actor
        torch.save(self.codebook.state_dict(), self.path_model + 'codebook_{}.pt'.format(epoch)) # codebook
        torch.save(self.c_learning.C.state_dict(), self.path_model + 'c_learning_{}.pt'.format(epoch)) # c_learning
        

    def plot(self,epoch):
        # colors
        colors = ['r', 'g', 'b', 'y', 'm', 'c', 'k', 'w']
        # create n_skill different and discriminable colors
        # colors = cm.rainbow(np.linspace(0, 1, self.n_skill))
        # colors = get_n_distinct_colors_deterministic(self.n_skill)
        # markers
        markers = ['o', '^', 's', '*', 'D']
        # set np array 
        self.rb_list_to_np(epoch)
        # s
        rb_shape = self.replay_buffer_states_np.shape
        s_right = self.replay_buffer_states_np.reshape(rb_shape[0] , rb_shape[1] * rb_shape[2], rb_shape[3])
        s_numpy = self.replay_buffer_states_np.reshape(rb_shape[0] * rb_shape[1] * rb_shape[2], rb_shape[3])
        s = torch.tensor(s_numpy, dtype=torch.float32).to(self.device)
        s0 = self.envs.envs[0].reset()[0]
        s0 = torch.tensor(s0, dtype=torch.float32).repeat(s.shape[0], 1).to(self.device)
        # su
        rb_shape_u = self.replay_buffer_s_u_np.shape
        s_u = self.replay_buffer_s_u_np.reshape(rb_shape_u[0] * rb_shape_u[1] * rb_shape_u[2], rb_shape_u[3])
        s_u = torch.tensor(s_u, dtype=torch.float32).to(self.device)
        # i 
        rb_shape = self.replay_buffer_infos_np.shape
        i = self.replay_buffer_infos_np.reshape(rb_shape[0] * rb_shape[1] * rb_shape[2], rb_shape[3])
        i_right = self.replay_buffer_infos_np.reshape(rb_shape[0] , rb_shape[1] * rb_shape[2], rb_shape[3])
        if self.env.name == 'fetch' or self.env.name == 'fetch_push' or self.env.name == 'finger' or self.env.name == 'hand':
            # pca 
            pca_b = True
            # pca 
            pca = PCA(n_components=2)
            pca.fit(i)
        else : 
            # pca
            pca_b = False
        with torch.no_grad():
            # main figure
            fig_main, ax_main = plt.subplots(1, 1 ,figsize=(10, 10)) 
            # measure figure
            rows = math.ceil(math.sqrt(self.n_skill))
            cols = math.ceil(self.n_skill / rows)
            fig_measure, ax_measure = plt.subplots(rows, cols , figsize=(10, 10)) 
            ax_measure = ax_measure.ravel() 
            # uncertainty figure
            fig_uncertainty, ax_uncertainty = plt.subplots(rows, cols, figsize=(10, 10))
            ax_uncertainty = ax_uncertainty.ravel()
            # reach figure
            fig_reach, ax_reach = plt.subplots(rows, cols, figsize=(10, 10))
            ax_reach = ax_reach.ravel()
            # measure, uncertainty, reach, state_archive  data 
            measure_data = []
            uncertainty_data = []
            reach_data = []
            state_archive_data = []
            info_archive_data = []  
            for idx in range(self.n_skill):
                # measure
                z_i = self.codebook.codebook_vectors[idx].repeat(s.shape[0], 1).to(self.device)
                a0, _, _= self.actor.get_action(s0, z_i)
                c = self.measure(s0, a0, s, z_i).cpu()
                measure = (c-c.min())/(c.max()-c.min())
                measure = measure.numpy()
                measure = list(measure.flatten())
                measure_data.append(measure)
                # uncertainty
                uncertainty_measure = self.uncertainty_measure(s, idx, epoch).cpu()
                uncertainty_measure = (uncertainty_measure - uncertainty_measure.min())/(uncertainty_measure.max()-uncertainty_measure.min())
                uncertainty_measure = list(uncertainty_measure.numpy().flatten())
                uncertainty_data.append(uncertainty_measure)
                # reach
                s_a_i = torch.tensor(self.archive[idx][-1]).repeat(s.shape[0], 1).to(self.device)
                m_reach = self.measure_fix(s, s_a_i, z_i, m =True).cpu()
                m_reach = (m_reach - m_reach.min())/(m_reach.max()-m_reach.min()).numpy()
                m_reach = list(m_reach.flatten())
                reach_data.append(m_reach)
                # info archive
                i_i =np.array(self.info_archive[idx])
                # i_i = list(i_i.flatten())
                info_archive_data.append(i_i)
                # state archive
                si =np.array(self.archive[idx])
                # si = list(si.flatten())
                state_archive_data.append(si)
            # plot 
            for idx in range(self.n_skill):
                if isinstance(self.env_class(), Maze): 
                    pca_b = False
                    # main
                    ax_main.scatter(s_right[idx, :, 0], s_right[idx, :, 1], c=colors[idx], label='skill {}'.format(idx), alpha=0.5)
                    # measure & title
                    sc = ax_measure[idx].scatter(list(i[:, 0].flatten()), list(i[:, 1].flatten()), c=measure_data[idx], cmap='viridis')
                    ax_measure[idx].set_title('skill {}'.format(idx))
                    # uncertainty & title
                    sc = ax_uncertainty[idx].scatter(list(i[:, 0].flatten()), list(i[:, 1].flatten()), c=uncertainty_data[idx], cmap='viridis')
                    ax_uncertainty[idx].set_title('skill {}'.format(idx))
                    # reach & title
                    sc = ax_reach[idx].scatter(list(i[:, 0].flatten()), list(i[:, 1].flatten()), c=reach_data[idx], cmap='viridis')
                    ax_reach[idx].set_title('skill {}'.format(idx))
                    # archive uncertainty & reach
                    for idx_2 in range(self.n_skill):
                        ax_uncertainty[idx_2].scatter(state_archive_data[idx_2][:, 0], state_archive_data[idx_2][:, 1], c=colors[idx_2], label='skill {}'.format(idx_2), alpha=1, s=150)
                        ax_reach[idx_2].scatter(state_archive_data[idx_2][:, 0], state_archive_data[idx_2][:, 1], c=colors[idx_2], label='skill {}'.format(idx_2), alpha=1, s=150)
                    # plot walls
                    for wall in self.envs.envs[0].walls:
                        x1, y1, x2, y2 = wall
                        ax_main.plot([x1, x2], [y1, y2], color='black')
                        ax_measure[idx].plot([x1, x2], [y1, y2], color='black')
                        ax_uncertainty[idx].plot([x1, x2], [y1, y2], color='black')
                        ax_reach[idx].plot([x1, x2], [y1, y2], color='black')
                elif self.env.name == 'fetch' or self.env.name == 'finger' or self.env.name == 'hand' :
                    # main
                    ax_main.scatter(pca.transform(i_right[idx, :, :])[:,0], pca.transform(i_right[idx, :, :])[:,1], c=colors[idx], label='skill {}'.format(idx), alpha=0.5)
                    # measure & title 
                    sc = ax_measure[idx].scatter(pca.transform(i)[:,0], pca.transform(i)[:,1], c=measure_data[idx], cmap='viridis')
                    ax_measure[idx].set_title('skill {}'.format(idx))
                    # uncertainty & title 
                    sc = ax_uncertainty[idx].scatter(pca.transform(i)[:,0], pca.transform(i)[:,1], c=uncertainty_data[idx], cmap='viridis')
                    ax_uncertainty[idx].set_title('skill {}'.format(idx))
                    # reach & title 
                    sc = ax_reach[idx].scatter(pca.transform(i)[:,0], pca.transform(i)[:,1], c=reach_data[idx], cmap='viridis')
                    ax_reach[idx].set_title('skill {}'.format(idx))
                    # archive uncertainty & reach
                    ax_uncertainty[idx].scatter(pca.transform(info_archive_data[idx])[:,0], pca.transform(info_archive_data[idx])[:,1], c=colors[idx], label='skill {}'.format(idx), alpha=1, s=150)
                    ax_reach[idx].scatter(pca.transform(info_archive_data[idx])[:,0], pca.transform(info_archive_data[idx])[:,1], c=colors[idx], label='skill {}'.format(idx), alpha=1, s=150)
                elif self.env.name == 'hopper' or self.env.name == 'half_cheetah' :
                    # main
                    ax_main.scatter(i_right[idx, :, -3], i_right[idx, :, -1], c=colors[idx], label='skill {}'.format(idx), alpha=0.5)
                    # measure & title
                    sc = ax_measure[idx].scatter(i[:, -3], i[:, -1], c=measure_data[idx], cmap='viridis')
                    ax_measure[idx].set_title('skill {}'.format(idx))
                    # uncertainty & title
                    sc = ax_uncertainty[idx].scatter(i[:, -3], i[:, -1], c=uncertainty_data[idx], cmap='viridis')
                    ax_uncertainty[idx].set_title('skill {}'.format(idx))
                    # reach & title
                    sc = ax_reach[idx].scatter(i[:, -3], i[:, -1], c=reach_data[idx], cmap='viridis')
                    ax_reach[idx].set_title('skill {}'.format(idx))
                    # archive uncertainty & reach
                    ax_uncertainty[idx].scatter(info_archive_data[idx][:, -3], info_archive_data[idx][:, -1], c=colors[idx], label='skill {}'.format(idx), alpha=1, s=150)
                    ax_reach[idx].scatter(info_archive_data[idx][:, -3], info_archive_data[idx][:, -1], c=colors[idx], label='skill {}'.format(idx), alpha=1, s=150)
                
                elif self.env.name == 'fetch_push' : 
                    if not self.featurize : 
                        # main
                        ax_main.scatter(pca.transform(i_right[idx, :, :])[:,0], pca.transform(i_right[idx, :, :])[:,1], c=colors[idx], label='skill {}'.format(idx), alpha=0.5)
                        # measure & title 
                        sc = ax_measure[idx].scatter(pca.transform(i)[:,0], pca.transform(i)[:,1], c=measure_data[idx], cmap='viridis')
                        ax_measure[idx].set_title('skill {}'.format(idx))
                        # uncertainty & title 
                        sc = ax_uncertainty[idx].scatter(pca.transform(i)[:,0], pca.transform(i)[:,1], c=uncertainty_data[idx], cmap='viridis')
                        ax_uncertainty[idx].set_title('skill {}'.format(idx))
                        # reach & title 
                        sc = ax_reach[idx].scatter(pca.transform(i)[:,0], pca.transform(i)[:,1], c=reach_data[idx], cmap='viridis')
                        ax_reach[idx].set_title('skill {}'.format(idx))
                        # archive uncertainty & reach
                        ax_uncertainty[idx].scatter(pca.transform(info_archive_data[idx])[:,0], pca.transform(info_archive_data[idx])[:,1], c=colors[idx], label='skill {}'.format(idx), alpha=1, s=150)
                        ax_reach[idx].scatter(pca.transform(info_archive_data[idx])[:,0], pca.transform(info_archive_data[idx])[:,1], c=colors[idx], label='skill {}'.format(idx), alpha=1, s=150)
                    else :
                        # main
                        ax_main.scatter(i_right[idx, :, 3], i_right[idx, :, 4], c=colors[idx], label='skill {}'.format(idx), alpha=0.5)
                        # measure & title
                        sc = ax_measure[idx].scatter(i[:, 3], i[:, 4], c=measure_data[idx], cmap='viridis')
                        ax_measure[idx].set_title('skill {}'.format(idx))
                        # uncertainty & title
                        sc = ax_uncertainty[idx].scatter(i[:, 3], i[:, 4], c=uncertainty_data[idx], cmap='viridis')
                        ax_uncertainty[idx].set_title('skill {}'.format(idx))
                        # reach & title
                        sc = ax_reach[idx].scatter(i[:, 3], i[:, 4], c=reach_data[idx], cmap='viridis')
                        ax_reach[idx].set_title('skill {}'.format(idx))
                        # archive uncertainty & reach
                        ax_uncertainty[idx].scatter(info_archive_data[idx][:, 3], info_archive_data[idx][:, 4], c=colors[idx], label='skill {}'.format(idx), alpha=1, s=150)
                        ax_reach[idx].scatter(info_archive_data[idx][:, 3], info_archive_data[idx][:, 4], c=colors[idx], label='skill {}'.format(idx), alpha=1, s=150)

                # else: 
                #     pca_b = True
                #     # pca 
                #     pca = PCA(n_components=2)
                #     pca.fit(s.cpu().numpy())
                #     # main
                #     ax_main.scatter(pca.transform(s_right[idx, :, :])[:,0], pca.transform(s_right[idx, :, :])[:,1], c=colors[idx], label='skill {}'.format(idx), alpha=0.5)
                #     # measure & title 
                #     sc = ax_measure[idx].scatter(pca.transform(s.cpu().numpy())[:,0], pca.transform(s.cpu().numpy())[:,1], c=measure_data[idx], cmap='viridis')
                #     ax_measure[idx].set_title('skill {}'.format(idx))
                #     # uncertainty & title 
                #     sc = ax_uncertainty[idx].scatter(pca.transform(s.cpu().numpy())[:,0], pca.transform(s.cpu().numpy())[:,1], c=uncertainty_data[idx], cmap='viridis')
                #     ax_uncertainty[idx].set_title('skill {}'.format(idx))
                #     # reach & title 
                #     sc = ax_reach[idx].scatter(pca.transform(s.cpu().numpy())[:,0], pca.transform(s.cpu().numpy())[:,1], c=reach_data[idx], cmap='viridis')
                #     ax_reach[idx].set_title('skill {}'.format(idx))
                #     # archive uncertainty & reach
                #     ax_uncertainty[idx].scatter(pca.transform(state_archive_data[idx])[:,0], pca.transform(state_archive_data[idx])[:,1], c=colors[idx], label='skill {}'.format(idx), alpha=1, s=150)
                #     ax_reach[idx].scatter(pca.transform(state_archive_data[idx])[:,0], pca.transform(state_archive_data[idx])[:,1], c=colors[idx], label='skill {}'.format(idx), alpha=1, s=150)
            
            # remove ticks, axis off and set limits
            # main 
            ax_main.set_xticks([]) if pca_b else None
            ax_main.set_yticks([]) if pca_b else None
            ax_main.axis('off') if pca_b else None 
            ax_main.set_xlim(self.env.min_x, self.env.max_x) if not pca_b else None
            ax_main.set_ylim(self.env.min_y, self.env.max_y) if not pca_b else None
            # measure
            for ax in ax_measure:
                ax.set_xticks([]) if pca_b else None
                ax.set_yticks([]) if pca_b else None
                ax.axis('off') if pca_b else None
                ax.set_xlim(self.env.min_x, self.env.max_x) if not pca_b else None
                ax.set_ylim(self.env.min_y, self.env.max_y) if not pca_b else None
            # uncertainty
            for ax in ax_uncertainty:
                ax.set_xticks([]) if pca_b else None
                ax.set_yticks([]) if pca_b else None
                ax.axis('off') if pca_b else None
                ax.set_xlim(self.env.min_x, self.env.max_x) if not pca_b else None
                ax.set_ylim(self.env.min_y, self.env.max_y) if not pca_b else None
            # reach
            for ax in ax_reach:
                ax.set_xticks([]) if pca_b else None
                ax.set_yticks([]) if pca_b else None
                ax.axis('off') if pca_b else None
                ax.set_xlim(self.env.min_x, self.env.max_x) if not pca_b else None
                ax.set_ylim(self.env.min_y, self.env.max_y) if not pca_b else None
            # colorbar
            cbar_ax = fig_measure.add_axes([0.925, 0.125, 0.025, 0.75])
            fig_measure.colorbar(sc, cax=cbar_ax)
            cbar_ax = fig_uncertainty.add_axes([0.925, 0.125, 0.025, 0.75])
            fig_uncertainty.colorbar(sc, cax=cbar_ax)
            # save figure
            fig_main.savefig(self.path_main+'epoch_{}.png'.format(epoch))
            fig_measure.savefig(self.path_measure+'epoch_{}.png'.format(epoch))
            fig_uncertainty.savefig(self.path_uncertainty+'epoch_{}.png'.format(epoch))
            fig_reach.savefig(self.path_reach+'epoch_{}.png'.format(epoch))
            # close fig 
            plt.close('all')
           

if __name__=="__main__":
    # from env.maze.maze import Maze
    # import time
    env_class=Maze    
    env_params = {'name': 'Hard'}
    n_c_r = 128
    featurize = True
    N = 40
    sigma = 0.01
    n_skill = 6
    z_dim = 20 
    lambda_d = 0.5
    lambda_a = 1.0
    beta = 1e5
    # lambda_e = 5e-2
    lambda_e = 0.1
    lambda_e_archive = 0.1
    lambda_o = 0.5
    gamma = 0.95
    lambda_clearning = 0.5
    lr_theta = 5e-4
    lr_skill = 5e-4
    lr_c = 1e-3
    n_episode = 16
    n_update = 16
    n_sgd_clearning = 1024
    # n_sgd_clearning = 64
    n_sgd_discriminable_loss = 16
    n_sgd_entropy_loss = 16
    n_archive = 1
    batch_size_clearning = 2**9*n_skill
    batch_size_loss = 2**9*n_skill
    seed = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    n_env = n_skill
    wandb_project_name = 'LDS'
    wandb_entity = 'lds_algorithm'
    t0= time.time()
    lds_algorithm = LDS( env_class, env_params, N, sigma, n_skill, z_dim, lambda_d, lambda_a, lambda_e, beta, lambda_e_archive, lambda_o, gamma, lambda_clearning, lr_theta, lr_skill, lr_c, n_episode,n_c_r, 
                n_update, n_sgd_clearning, n_sgd_discriminable_loss, n_sgd_entropy_loss, n_archive, batch_size_clearning, batch_size_loss, device, n_env, wandb_project_name, wandb_entity, seed, featurize)
    print('Initializing time : ', time.time()-t0)
    lds_algorithm.train()

    # # Fetch
    # env_class= WFetch   
    # env_params = {'name':'fetch'}
    # n_c_r = 128
    # featurize = True
    # N = 20
    # sigma = 0.01
    # n_skill = 6
    # z_dim = 20 
    # lambda_d = 0.5
    # lambda_a = 1.0
    # beta = 1e5
    # lambda_e = 0.01
    # lambda_e_archive = 0.01
    # lambda_o = 0.5
    # gamma = 0.95
    # lambda_clearning = 0.5
    # lr_theta = 5e-4
    # lr_skill = 5e-4
    # lr_c = 1e-3
    # n_episode = 8
    # n_update = 16
    # n_sgd_clearning = 1024
    # n_sgd_discriminable_loss = 16
    # n_sgd_entropy_loss = 16
    # n_archive = 1
    # batch_size_clearning = 2**8*(n_skill)
    # batch_size_loss = 2**8*(n_skill)
    # seed = 0
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # # device = 'cpu'
    # n_env = n_skill
    # wandb_project_name = 'LDS'
    # wandb_entity = 'lds_algorithm'
    # t0= time.time()
    # lds_algorithm = LDS( env_class, env_params, N, sigma, n_skill, z_dim, lambda_d, lambda_a, lambda_e, beta, lambda_e_archive, lambda_o, gamma, lambda_clearning, lr_theta, lr_skill, lr_c, n_episode, n_c_r,
    #             n_update, n_sgd_clearning, n_sgd_discriminable_loss, n_sgd_entropy_loss, n_archive, batch_size_clearning, batch_size_loss, device, n_env, wandb_project_name, wandb_entity, seed, featurize)
    # print('Initializing time : ', time.time()-t0)
    # lds_algorithm.train()

    # Hand
    # env_class= WHand    
    # env_params = {'name': 'hand'}
    # n_c_r = 64
    # featurize = True
    # N = 50
    # sigma = 0.01
    # n_skill = 12
    # z_dim = 20 
    # lambda_d = 0.5
    # lambda_a = 1.0
    # beta = 1e5
    # lambda_e = 0.0
    # lambda_e_archive = 0.001
    # lambda_o = 0.05
    # gamma = 0.95
    # lambda_clearning = 0.5
    # lr_theta = 1e-3
    # lr_skill = 5e-4
    # lr_c = 1e-3
    # n_episode = 8
    # n_update = 32
    # n_sgd_clearning = 1024
    # n_sgd_discriminable_loss = 16
    # n_sgd_entropy_loss = 16
    # n_archive = 1
    # batch_size_clearning = 2**8*(n_skill)
    # batch_size_loss = 2**8*(n_skill)
    # seed = 0
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # # device = 'cpu'
    # n_env = n_skill
    # wandb_project_name = 'LDS'
    # wandb_entity = 'lds_algorithm'
    # t0= time.time()
    # lds_algorithm = LDS( env_class, env_params, N, sigma, n_skill, z_dim, lambda_d, lambda_a, lambda_e, beta, lambda_e_archive, lambda_o, gamma, lambda_clearning, lr_theta, lr_skill, lr_c, n_episode, n_c_r,
    #             n_update, n_sgd_clearning, n_sgd_discriminable_loss, n_sgd_entropy_loss, n_archive, batch_size_clearning, batch_size_loss, device, n_env, wandb_project_name, wandb_entity, seed, featurize)
    # print('Initializing time : ', time.time()-t0)
    # lds_algorithm.train()

    # # Hopper
    # env_class=Whopper
    # env_params = {'name': 'hopper'}
    # n_c_r = 128
    # featurize = True
    # N = 50
    # sigma = 0.01
    # n_skill = 6
    # z_dim = 20 
    # lambda_d = 0.5
    # lambda_a = 1.0
    # beta = 1e5
    # # lambda_e = 5e-2
    # lambda_e = 0.1
    # lambda_e_archive = 0.0
    # lambda_o = 0.05
    # gamma = 0.95
    # lambda_clearning = 0.5
    # lr_theta = 5e-4
    # lr_skill = 5e-4
    # lr_c = 1e-3
    # n_episode = 16
    # n_update = 16
    # n_sgd_clearning = 1024
    # n_sgd_discriminable_loss = 16
    # n_sgd_entropy_loss = 16
    # n_archive = 1
    # batch_size_clearning = 2**8*(n_skill)
    # batch_size_loss = 2**8*(n_skill)
    # seed = 0
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # # device = 'cpu'
    # # device = 'cpu'
    # n_env = n_skill
    # wandb_project_name = 'LDS'
    # wandb_entity = 'lds_algorithm'
    # t0= time.time()
    # lds_algorithm = LDS( env_class, env_params, N, sigma, n_skill, z_dim, lambda_d, lambda_a, lambda_e, beta, lambda_e_archive, lambda_o, gamma, lambda_clearning, lr_theta, lr_skill, lr_c, n_episode, n_c_r, 
    #             n_update, n_sgd_clearning, n_sgd_discriminable_loss, n_sgd_entropy_loss, n_archive, batch_size_clearning, batch_size_loss, device, n_env, wandb_project_name, wandb_entity, seed, featurize)
    # print('Initializing time : ', time.time()-t0)
    # lds_algorithm.train()

    # # Fetch_Push
    # env_class= WFetch_Push
    # env_params = {'name':'fetch_push'}
    # n_c_r = 256
    # N = 50
    # sigma = 0.01
    # n_skill = 6
    # z_dim = 20 
    # lambda_d = 0.5
    # lambda_a = 1.0
    # beta = 1e5
    # featurize = True
    # # lambda_e = 5e-2
    # lambda_e = 0.005
    # lambda_e_archive = 0.005
    # lambda_o = 0.05
    # gamma = 0.95
    # lambda_clearning = 0.5
    # lr_theta = 5e-4
    # lr_skill = 5e-4
    # lr_c = 1e-3
    # n_episode = 8
    # n_update = 16
    # n_sgd_clearning = 2048
    # # n_sgd_clearning = 64
    # n_sgd_discriminable_loss = 16
    # n_sgd_entropy_loss = 16
    # n_archive = 1
    # batch_size_clearning = 2**8*(n_skill)
    # batch_size_loss = 2**8*(n_skill)
    # seed = 0
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # # device = 'cpu'
    # n_env = n_skill
    # wandb_project_name = 'LDS'
    # wandb_entity = 'lds_algorithm'
    # t0= time.time()
    # lds_algorithm = LDS( env_class, env_params, N, sigma, n_skill, z_dim, lambda_d, lambda_a, lambda_e, beta, lambda_e_archive, lambda_o, gamma, lambda_clearning, lr_theta, lr_skill, lr_c, n_episode, n_c_r,
    #             n_update, n_sgd_clearning, n_sgd_discriminable_loss, n_sgd_entropy_loss, n_archive, batch_size_clearning, batch_size_loss, device, n_env, wandb_project_name, wandb_entity, seed, featurize)
    # print('Initializing time : ', time.time()-t0)
    # lds_algorithm.train()

    # # Wfinger
    # env_class= WFinger
    # env_params = {'name':'finger'}
    # n_c_r = 128
    # featurize = True
    # N = 50
    # sigma = 0.01
    # n_skill = 6
    # z_dim = 20 
    # lambda_d = 0.5
    # lambda_a = 1.0
    # beta = 1e5
    # # lambda_e = 5e-2
    # lambda_e = 0.05
    # lambda_e_archive = 0.01
    # lambda_o = 0.05
    # gamma = 0.95
    # lambda_clearning = 0.5
    # lr_theta = 5e-4
    # lr_skill = 5e-4
    # lr_c = 1e-3
    # n_episode = 8
    # n_update = 16
    # n_sgd_clearning = 1024
    # # n_sgd_clearning = 64
    # n_sgd_discriminable_loss = 16
    # n_sgd_entropy_loss = 16
    # n_archive = 1
    # batch_size_clearning = 2**9*(n_skill)
    # batch_size_loss = 2**9*(n_skill)
    # seed = 0
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # # device = 'cpu'
    # n_env = n_skill
    # wandb_project_name = 'LDS'
    # wandb_entity = 'lds_algorithm'
    # t0= time.time()
    # lds_algorithm = LDS( env_class, env_params, N, sigma, n_skill, z_dim, lambda_d, lambda_a, lambda_e, beta, lambda_e_archive, lambda_o, gamma, lambda_clearning, lr_theta, lr_skill, lr_c, n_episode,n_c_r, 
    #             n_update, n_sgd_clearning, n_sgd_discriminable_loss, n_sgd_entropy_loss, n_archive, batch_size_clearning, batch_size_loss, device, n_env, wandb_project_name, wandb_entity, seed)
    # print('Initializing time : ', time.time()-t0)
    # lds_algorithm.train()

    # # halfcheetah
    # env_class=Whalfcheetah
    # env_params = {'name': 'halfcheetah'}
    # n_c_r = 128
    # featurize = False
    # N = 50
    # sigma = 0.01
    # n_skill = 6
    # z_dim = 20 
    # lambda_d = 0.5
    # lambda_a = 1.0
    # beta = 1e5
    # # lambda_e = 5e-2
    # lambda_e = 0.01
    # lambda_e_archive = 0.0
    # lambda_o = 0.05
    # gamma = 0.95
    # lambda_clearning = 0.5
    # lr_theta = 5e-4
    # lr_skill = 5e-4
    # lr_c = 1e-3
    # n_episode = 8
    # n_update = 16
    # n_sgd_clearning = 1024
    # n_sgd_discriminable_loss = 16
    # n_sgd_entropy_loss = 16
    # n_archive = 1
    # batch_size_clearning = 2**8*(n_skill)
    # batch_size_loss = 2**8*(n_skill)
    # seed = 0
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # # device = 'cpu'
    # # device = 'cpu'
    # n_env = n_skill
    # wandb_project_name = 'LDS'
    # wandb_entity = 'lds_algorithm'
    # t0= time.time()
    # lds_algorithm = LDS( env_class, env_params, N, sigma, n_skill, z_dim, lambda_d, lambda_a, lambda_e, beta, lambda_e_archive, lambda_o, gamma, lambda_clearning, lr_theta, lr_skill, lr_c, n_episode, n_c_r, 
    #             n_update, n_sgd_clearning, n_sgd_discriminable_loss, n_sgd_entropy_loss, n_archive, batch_size_clearning, batch_size_loss, device, n_env, wandb_project_name, wandb_entity, seed, featurize)
    # print('Initializing time : ', time.time()-t0)
    # lds_algorithm.train()
