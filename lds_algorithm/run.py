import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import os
import random
import time
from distutils.util import strtobool
from env.maze.maze import Maze
from env.mujoco.whand import WHand
from env.mujoco.wfetch import WFetch
from env.mujoco.whopper import Whopper
from env.mujoco.wfinger import WFinger
from env.mujoco.wfetch_push import WFetch_Push
import gym 
from lds_a import LDS


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_class', type=str, default='hopper')
    parser.add_argument('--env_name', type=str, default='hopper')
    parser.add_argument('--featurize', type=strtobool, default=False)
    parser.add_argument('--n_c_r', type=int, default=128)
    parser.add_argument('--N', type=int, default=150)
    parser.add_argument('--sigma', type=float, default=0.01)
    parser.add_argument('--n_skill', type=int, default=6)
    parser.add_argument('--z_dim', type=int, default=20)
    parser.add_argument('--lambda_d', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=1e5)
    parser.add_argument('--lambda_e', type=float, default=5e-3)
    parser.add_argument('--lambda_a', type=float, default=1.0)
    parser.add_argument('--lambda_e_archive', type=float, default=0.5)
    parser.add_argument('--lambda_o', type=float, default=0.05)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lambda_clearning', type=float, default=0.4)
    parser.add_argument('--lr_theta', type=float, default=2e-4)
    parser.add_argument('--lr_skill', type=float, default=5e-5)
    parser.add_argument('--lr_c', type=float, default=5e-4)
    parser.add_argument('--n_episode', type=int, default=8)
    parser.add_argument('--n_update', type=int, default=32)
    parser.add_argument('--n_sgd_clearning', type=int, default=256)
    parser.add_argument('--n_sgd_discriminable_loss', type=int, default=16)
    parser.add_argument('--n_sgd_entropy_loss', type=int, default=16)
    parser.add_argument('--n_archive', type=int, default=1)
    parser.add_argument('--batch_size_clearning', type=int, default=1024)
    parser.add_argument('--batch_size_loss', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--n_env', type=int, default=6)
    parser.add_argument('--wandb_project_name', type=str, default='LDS')
    parser.add_argument('--wandb_entity', type=str, default='lds_algorithm')
    args = parser.parse_args()
    # fmt: on
    return args

args = parse_args()
if args.env_class == 'maze':
    env_class = Maze
elif args.env_class == 'hopper':
    env_class = Whopper
elif args.env_class == 'hand':
    env_class = WHand
elif args.env_class == 'fetch':
    env_class = WFetch
elif args.env_class == 'finger':
    env_class = WFinger
elif args.env_class == 'fetch_push':
    env_class = WFetch_Push
    
algo = LDS(env_class=env_class, env_params={'name':args.env_name}, N=args.N, sigma=args.sigma, 
           n_skill=args.n_skill, z_dim=args.z_dim, lambda_d=args.lambda_d, beta=args.beta, 
           lambda_e=args.lambda_e, lambda_a=args.lambda_a, lambda_e_archive=args.lambda_e_archive, lambda_o=args.lambda_o, 
           gamma=args.gamma, lambda_clearning=args.lambda_clearning, lr_theta=args.lr_theta, lr_skill=args.lr_skill, 
           lr_c=args.lr_c, n_episode=args.n_episode, n_update=args.n_update, n_sgd_clearning=args.n_sgd_clearning, 
           n_sgd_discriminable_loss=args.n_sgd_discriminable_loss, n_sgd_entropy_loss=args.n_sgd_entropy_loss, 
           n_archive=args.n_archive, batch_size_clearning=args.batch_size_clearning, batch_size_loss=args.batch_size_loss, 
           seed=args.seed, device=args.device, n_env=args.n_env, wandb_project_name=args.wandb_project_name, wandb_entity=args.wandb_entity, featurize=args.featurize, n_c_r=args.n_c_r)
algo.train()

