import os, sys
# from utils.wrapper import CustomEnvWrapper, make_env
from utils.tools import aplatir
from utils.wandb_server import WandbServer
from utils.sac import Actor
from utils.codebook import Codebook
from sklearn.decomposition import PCA
from env.maze.maze import Maze
# from env.mujoco.whand import WHand
# from env.mujoco.wfetch import WFetch
from env.mujoco.whopper import Whopper
# from env.mujoco.wfinger import WFinger
# from env.mujoco.wfetch_push import WFetch_Push
import copy
import math
import gym
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import wandb
import random



class Algorithm(object):
    def __init__(self, algo, env_class, env_params,  N, n_skill, z_dim, gamma, lr_theta, n_episode,  batch_size, device, n_env, wandb_project_name, wandb_entity, seed, featurize): 
        """ Initialize thealgorithm
        N: number of epochs
        n_skill: number of skills
        z_dim: dimension of the skill space
        gamma: discount factor
        lr_theta: learning rate of the actor
        lr_skill: learning rate of the codebook
        lr_c: learning rate of the successor state measure
        batch_size: batch size for the replay buffer
        device: device to use for the computations
        n_env: number of parallel environments"""
        self.featurize = featurize
        self.seed = seed
        self.set_all_seeds(seed)
        self.algo = algo
        self.env_class = env_class
        self.env = env_class(**env_params) #To be changed for other environments
        self.N = N
        self.n_skill = n_skill
        self.n_episode = n_episode
        self.z_dim = z_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = device
        self.n_env = n_env
        self.wandb_project_name = wandb_project_name
        self.wandb_entity = self.algo+'_'+self.env.name+'_'+str(self.seed)
        self.wandb_server = WandbServer(self.wandb_project_name, self.wandb_entity)
        # replay_buffer 
        self.replay_buffer_states = [[[]] for i in range(n_skill)]
        self.replay_buffer_actions = [[[]] for i in range(n_skill)]
        self.replay_buffer_rewards = [[[]] for i in range(n_skill)]
        self.replay_buffer_infos = [[[]] for i in range(n_skill)]
        self.replay_buffer_dones = [[[]] for i in range(n_skill)]
        self.replay_buffer_z = [[] for i in range(n_skill)]
        # replay buffer np array
        self.replay_buffer_states_np = None
        self.replay_buffer_actions_np = None
        self.replay_buffer_z_np = None
        # envs 
        self.envs = gym.vector.SyncVectorEnv([lambda : env_class(**env_params) for i in range(n_env)]) #To be changed for other environments
        # makedir 
        self.path = '../runs/'+self.algo+'/'+self.env.name+'/'+'seed_'+str(self.seed)+'/'
        self.path_main = self.path+'/main/'
        self.path_model = self.path+'/model/'
        os.makedirs(self.path, exist_ok=True)
        os.makedirs(self.path_main, exist_ok=True)
        os.makedirs(self.path_model, exist_ok=True)
        # type 
        self.maze_type = True if self.env.name == 'Easy' or self.env.name == 'Ur' or self.env.name == 'Hard' else False
        # coverage maze 
        self.accuracy = 10
        if self.maze_type : 
            self.map = np.zeros((self.accuracy, self.accuracy)) #XY
            self.ratio = 1
        elif self.env.name == 'fetch' : 
            self.map = np.zeros((self.accuracy, self.accuracy, self.accuracy)) #XYZ Gripper
            self.ratio = 1
        elif self.env.name=='fetch_push' : 
            self.map =np.zeros((self.accuracy, self.accuracy, self.accuracy, self.accuracy, self.accuracy)) if not featurize else np.zeros((self.accuracy, self.accuracy)) # X_puck Y_puck
            self.ratio = 1
        elif self.env.name == 'hopper' : 
            self.map =np.zeros((self.accuracy, self.accuracy)) #XY foot
            self.ratio = 1
        elif self.env.name == 'finger' : 
            self.map =np.zeros((self.accuracy, self.accuracy, self.accuracy)) #theta1_f theta2_f theta1_d
            self.ratio = 1
        else : 
            self.map =np.zeros((self.accuracy, self.accuracy)) 
            self.ratio = 1


        # sample 
        self.sample_nb = 0
        


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
                z_omega=self.codebook.codebook_vectors.detach()
                s , i= self.envs.reset()
                for z_idx in range(self.n_skill) : self.replay_buffer_states[z_idx][-1].append(s[z_idx])
                for z_idx in range(self.n_skill) : self.replay_buffer_z[z_idx].append(z_omega[z_idx].cpu().numpy())
                for t in range(self.env.max_steps):
                    with torch.no_grad():
                        a, _, mean= self.actor.get_action(torch.tensor(s, dtype=torch.float32, device=self.device), z_omega)
                        a = a.detach().cpu().numpy()
                        s,r,d,t,i= self.envs.step(a)
                        for z_idx in range(self.n_skill) : self.replay_buffer_actions[z_idx][-1].append(a[z_idx])
                        for z_idx in range(self.n_skill) : self.replay_buffer_states[z_idx][-1].append(s[z_idx])
                       
                # add new list for next sampling
                for z_idx in range(self.n_skill) : self.replay_buffer_states[z_idx].append([])
                for z_idx in range(self.n_skill) : self.replay_buffer_actions[z_idx].append([])

    def update_coverage(self,s,z_idx):
        if self.maze_type : 
            x,y = s[z_idx][0], s[z_idx][1]
            x = (x-self.env.min_x)/(self.env.max_x - self.env.min_x)
            y = (y-self.env.min_y)/(self.env.max_y - self.env.min_y)
            x_coord = np.clip(int(x*self.map.shape[0]),0,self.map.shape[0]-1)
            y_coord = np.clip(int(y*self.map.shape[1]),0,self.map.shape[1]-1)
            self.map[x_coord,y_coord]=1
        elif self.env.name == 'fetch' : 
            x,y,z = s[z_idx][0], s[z_idx][1], s[z_idx][2]
            x = (x-self.env.min_x)/(self.env.max_x - self.env.min_x)
            y = (y-self.env.min_y)/(self.env.max_y - self.env.min_y)
            z = (z-self.env.min_z)/(self.env.max_z - self.env.min_z)
            x_coord = np.clip(int(x*self.map.shape[0]),0,self.map.shape[0]-1)
            y_coord = np.clip(int(y*self.map.shape[1]),0,self.map.shape[1]-1)
            z_coord = np.clip(int(z*self.map.shape[1]),0,self.map.shape[1]-1)
            self.map[x_coord,y_coord, z_coord]=1
        elif self.env.name == 'hopper' : 
            # x,y = s[z_idx][0], s[z_idx][2]
            x,y = s[z_idx][-3], s[z_idx][-1]
            x = (x-self.env.min_x)/(self.env.max_x - self.env.min_x)
            y = (y-self.env.min_y)/(self.env.max_y - self.env.min_y)
            x_coord = np.clip(int(x*self.map.shape[0]),0,self.map.shape[0]-1)
            y_coord = np.clip(int(y*self.map.shape[1]),0,self.map.shape[1]-1)
            self.map[x_coord,y_coord]=1
        elif self.env.name == 'fetch_push' : 
            x,y,z, x_puck, y_puck= s[z_idx][0], s[z_idx][1], s[z_idx][2], s[z_idx][3], s[z_idx][4]
            if self.featurize : 
                x = (x_puck-self.env.min_x)/(self.env.max_x - self.env.min_x)
                y = (y_puck-self.env.min_y)/(self.env.max_y - self.env.min_y)
                x_coord = np.clip(int(x*self.map.shape[0]),0,self.map.shape[0]-1)
                y_coord = np.clip(int(y*self.map.shape[1]),0,self.map.shape[1]-1)
                self.map[x_coord,y_coord]=1
            else :
                x = (x-self.env.min_x)/(self.env.max_x - self.env.min_x)
                y = (y-self.env.min_y)/(self.env.max_y - self.env.min_y)
                z = (z-self.env.min_z)/(self.env.max_z - self.env.min_z)
                x_puck = (x_puck-self.env.min_puck_x)/(self.env.max_puck_x - self.env.min_puck_x)
                y_puck = (y_puck-self.env.min_puck_y)/(self.env.max_puck_y - self.env.min_puck_y)
                x_coord = np.clip(int(x*self.map.shape[0]),0,self.map.shape[0]-1)
                y_coord = np.clip(int(y*self.map.shape[1]),0,self.map.shape[1]-1)
                z_coord = np.clip(int(z*self.map.shape[1]),0,self.map.shape[1]-1)
                x_puck_coord = np.clip(int(x_puck*self.map.shape[0]),0,self.map.shape[0]-1)
                y_puck_coord = np.clip(int(y_puck*self.map.shape[1]),0,self.map.shape[1]-1)
                self.map[x_coord,y_coord, z_coord, x_puck_coord, y_puck_coord]=1
        elif self.env.name == 'finger' : 
            theta_1, theta_2, theta_3 = s[z_idx][0], s[z_idx][1], s[z_idx][2]
            x = (theta_1-self.env.min_x)/(self.env.max_x - self.env.min_x)
            y = (theta_2-self.env.min_y)/(self.env.max_y - self.env.min_y)
            z = (theta_3-self.env.min_z)/(self.env.max_z - self.env.min_z)
            x_coord = np.clip(int(x*self.map.shape[0]),0,self.map.shape[0]-1)
            y_coord = np.clip(int(y*self.map.shape[1]),0,self.map.shape[1]-1)
            z_coord = np.clip(int(z*self.map.shape[1]),0,self.map.shape[1]-1)
            self.map[x_coord,y_coord, z_coord]=1

    def sample(self,batch_size, rho_stationary = False):
        """ Sample a batch of transitions from the replay buffer """
        # sample skills 
        idx_z = np.random.choice(np.arange(self.codebook.n), size=(batch_size, 1))
        z = self.codebook.codebook_vectors[idx_z[:,0]]
        s = []
        for idx in idx_z[:, 0]:
            # s1
            idx_episode = np.random.randint(0,len(self.replay_buffer_z[idx]))
            idx_time_step= np.random.randint(0,len(self.replay_buffer_states[idx][idx_episode])-1)
            # add batch 
            s.append(torch.tensor(self.replay_buffer_states[idx][idx_episode][idx_time_step]))
        s = torch.stack(s,dim=0).to(self.device)
        return(s, z, idx_z)
    
    def train(self):
        pass

    def plot_maze(self,epoch):
        # set np array 
        self.rb_list_to_np(epoch)
        # plot
        colors = ['r', 'g', 'b', 'y', 'm', 'c', 'k', 'w']
        cmap = plt.cm.get_cmap('viridis')
        colorbars = []
        fig, ax1 = plt.subplots(1, 1 ,figsize=(10, 10))       
        s0 = self.envs.envs[0].reset()[0]
        rb_shape = self.replay_buffer_states_np.shape
        s_right = self.replay_buffer_states_np.reshape(rb_shape[0] , rb_shape[1] * rb_shape[2], rb_shape[3])
        for i in range(self.n_skill):
            # ax1
            ax1.scatter(s_right[i, :, 0], s_right[i, :, 1], c=colors[i], label='skill {}'.format(i), alpha=0.5)

        for wall in self.envs.envs[0].walls:
                x1, y1, x2, y2 = wall
                ax1.plot([x1, x2], [y1, y2], color='black')
        ax1.set_xlim([-self.env.max_x, self.env.max_x])
        ax1.set_ylim([-self.env.max_y, self.env.max_y])
        ax1.set_axis_off()
        ax1.set_xticks([])
        ax1.set_yticks([])
            # ax1.legend()
        # save figure
        fig.savefig(self.path_main+'epoch_{}.png'.format(epoch))
        # close fig 
        plt.close('all')
            

    def rb_list_to_np(self,epoch):
        list_rb_states = copy.deepcopy(self.replay_buffer_states)
        list_rb_actions = copy.deepcopy(self.replay_buffer_actions)
        list_rb_infos = copy.deepcopy(self.replay_buffer_infos)
        list_rb_z = copy.deepcopy(self.replay_buffer_z)
        for i in range(len(list_rb_states)): 
            list_rb_states[i]=list_rb_states[i][-self.n_episode-1:-1]
            list_rb_actions[i]=list_rb_actions[i][-self.n_episode-1:-1]
            list_rb_infos[i]=list_rb_infos[i][-self.n_episode-1:-1]
            list_rb_z[i]=list_rb_z[i][-self.n_episode-1:-1]
        self.replay_buffer_z_np = np.array(list_rb_z)
        self.replay_buffer_states_np = np.array(list_rb_states)
        self.replay_buffer_actions_np = np.array(list_rb_actions)
        self.replay_buffer_infos_np = np.array(list_rb_infos)
       

    def coverage(self) : 
        return (np.sum(self.map) / self.map.size) * 100 * self.ratio
    

    def plot_mujoco(self,epoch):
        if self.algo == 'diayn' or self.algo == 'lsd':
            fig, ax1 = plt.subplots(1, 1 ,figsize=(10, 10))
            colors = ['r', 'g', 'b', 'y', 'm', 'c', 'k', 'w']
            # set np array 
            self.rb_list_to_np(epoch)
            # s
            rb_shape_s = self.replay_buffer_states_np.shape
            s_right = self.replay_buffer_states_np.reshape(rb_shape_s[0] , rb_shape_s[1] * rb_shape_s[2], rb_shape_s[3])
            s_numpy = self.replay_buffer_states_np.reshape(rb_shape_s[0] * rb_shape_s[1] * rb_shape_s[2], rb_shape_s[3])
            s = torch.tensor(s_numpy, dtype=torch.float32).to(self.device)
            # i 
            rb_shape_i = self.replay_buffer_infos_np.shape
            i = self.replay_buffer_infos_np.reshape(rb_shape_i[0] * rb_shape_i[1] * rb_shape_i[2], rb_shape_i[3])
            i_right = self.replay_buffer_infos_np.reshape(rb_shape_i[0] , rb_shape_i[1] * rb_shape_i[2], rb_shape_i[3])
            if isinstance(self.env_class(), Maze): 
                # plot
                for idx in range(self.n_skill):
                    # ax1
                    ax1.scatter(s_right[idx, :, 0], s_right[idx, :, 1], c=colors[idx], label='skill {}'.format(idx), alpha=0.5)
                for wall in self.envs.envs[0].walls:
                    x1, y1, x2, y2 = wall
                    ax1.plot([x1, x2], [y1, y2], color='black')
                ax1.set_xlim([-self.env.max_x, self.env.max_x])
                ax1.set_ylim([-self.env.max_y, self.env.max_y])

            elif self.env.name == 'fetch_push' :
                # plot
                for idx in range(self.n_skill):
                    # ax1
                    ax1.scatter(s_right[idx, :, 3], s_right[idx, :, 4], c=colors[idx], label='skill {}'.format(idx), alpha=0.5)
                
            else : 
                # pca 
                pca = PCA(n_components=2)
                pca.fit(i)
                for idx in range(self.n_skill):
                    # main
                    ax1.scatter(pca.transform(i_right[idx, :, :])[:,0], pca.transform(i_right[idx, :, :])[:,1], c=colors[idx], label='skill {}'.format(idx), alpha=0.5)
                
            ax1.set_axis_off()
            ax1.set_xticks([])
            ax1.set_yticks([])
            # ax1.legend()
            # save figure
            fig.savefig(self.path_main+'epoch_{}.png'.format(epoch))
            # close fig 
            plt.close('all')
            
            # ax1.legend()
            # set axis 
            # ax1.set_axis_off()
            # ax1.set_xticks([])
            # ax1.set_yticks([])
            # save figure
            fig.savefig(self.path_main+'epoch_{}.png'.format(epoch))
            # close fig 
            plt.close('all')

    def save_model(self, epoch):
        # save model
        torch.save(self.actor.state_dict(), self.path_model + 'actor_{}.pt'.format(epoch)) # actor
        torch.save(self.codebook.state_dict(), self.path_model + 'codebook_{}.pt'.format(epoch)) # codebook
