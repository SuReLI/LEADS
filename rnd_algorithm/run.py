import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import os
import random
import time
from distutils.util import strtobool
from env.maze.maze import Maze
from env.mujoco.wfetch import WFetch
from env.mujoco.wfetch_push import WFetch_Push
from env.mujoco.wfinger import WFinger
from env.mujoco.whopper import Whopper
from env.mujoco.whand import WHand
import gym 
from rnd_a import RND_A
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_class', type=str, default='fetch_push')
    parser.add_argument('--env_name', type=str, default='fetch_push')
    parser.add_argument('--N', type=int, default=500)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr_theta', type=float, default=5e-4)
    parser.add_argument('--lr_q', type=float, default=1e-3)
    parser.add_argument('--lr_starts', type=float, default=5e2)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--policy_frequency', type=int, default=2)
    parser.add_argument('--rnd_frequency', type=int, default=5)
    parser.add_argument('--target_frequency', type=int, default=1)
    parser.add_argument('--noise', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--autotune', type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument('--n_episode', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--n_env', type=int, default=5)
    parser.add_argument('--wandb_project_name', type=str, default='LDS')
    parser.add_argument('--wandb_entity', type=str, default='rnd_algorithm')
    args = parser.parse_args()
    return args

args = parse_args()
if args.env_class == 'maze':
    env_class = Maze
elif args.env_class == 'hopper':
    env_class = Whopper
elif args.env_class == 'fetch':
    env_class = WFetch
elif args.env_class == 'finger':
    env_class = WFinger
elif args.env_class == 'fetch_push':
    env_class = WFetch_Push
    
algo = RND_A(algo='rnd', env_class=env_class, env_params={'name':args.env_name}, N=args.N, 
            gamma=args.gamma, lr_theta=args.lr_theta, lr_q=args.lr_q, lr_starts=args.lr_starts, 
            tau=args.tau, policy_frequency=args.policy_frequency, rnd_frequency = args.rnd_frequency,  target_frequency=args.target_frequency, noise=args.noise, 
            alpha=args.alpha, autotune=args.autotune, n_episode=args.n_episode, batch_size=args.batch_size, seed=args.seed, 
            device=args.device, n_env=args.n_env, wandb_project_name=args.wandb_project_name, wandb_entity=args.wandb_entity)
algo.train()

