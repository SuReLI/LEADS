import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.mujoco.whopper import Whopper
from env.mujoco.whand import WHand
from env.mujoco.wfinger import WFinger
from env.mujoco.wfetch import WFetch
from env.mujoco.wfetch_push import WFetch_Push
import matplotlib.pyplot as plt
from lds_a import LDS 
import torch 
import time
from codebook import Codebook
from sac import Actor
import random
import numpy as np 
sigma = 0.01
n_skill = 12
z_dim = 20 
lambda_clearning = 0.5
lr_theta = 5e-4
lr_skill = 5e-4
seed = 0
n_step = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random.seed(seed)          # Python
np.random.seed(seed)       # Numpy
torch.manual_seed(seed)    # PyTorch
# Si vous utilisez CUDA pour PyTorch
# if torch.cuda.is_available():
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# path = '../runs/lds/hopper/seed_0/model'
# path = '../runs/lds/finger/seed_0/model'
# path = '../runs/lds/fetch_push/seed_0/model'
path = '../runs/lds/hand/seed_0/model'
model_number = 20
# env_class = Whopper
# env_class = WFinger
# env_class = WFetch_Push
env_class = WHand
env_unwrapped = env_class(render=True, render_mode='rgb_array')
# env_unwrapped = WHand(render=True)
codebook = Codebook(n_skill, z_dim, device, std=sigma, learning_rate=lr_skill)
codebook.load_state_dict(torch.load(path+'/codebook_'+str(model_number)+'.pt'))
actor = Actor(env_unwrapped.observation_space.shape[0], env_unwrapped.action_space.shape[0], z_dim,  env_unwrapped.action_space).to(device)
actor.load_state_dict(torch.load(path+'/actor_'+str(model_number)+'.pt'))

for i in range(n_skill):
    env =env_class(render=True)
    s = env.reset()[0] 
    for step in range(env_unwrapped.max_steps) : 
        with torch.no_grad() : 
            # input()
            a_s = actor.get_action(torch.tensor(s, dtype = torch.float32).unsqueeze(0).to(device), codebook.codebook_vectors[i].unsqueeze(0))[-1][0].cpu().numpy()
        s = env.step(a_s)[0]
        img = env.render()
    # save last image for each skill
    plt.imsave('./img_skills/skill_'+str(i)+'.png', img)
    env.close()
