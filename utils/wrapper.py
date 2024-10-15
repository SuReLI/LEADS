import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gym
import numpy as np
from gym import spaces

def make_env(env_class, env_id, z_dim, seed, idx, capture_video, run_name):
    def thunk():
        env = env_class()
        env = CustomEnvWrapper(env,z_dim)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

class CustomEnvWrapper(gym.Wrapper):
    def __init__(self, env, z_dim):
        super().__init__(env)
        self.z_dim = z_dim
        # Extend observation space with z_dim
        low = np.concatenate([self.observation_space.low, np.full(z_dim, -np.inf)])
        high = np.concatenate([self.observation_space.high, np.full(z_dim, np.inf)])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.z=None

    def set_z(self,z):
        self.z=z

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # Concatenate z to observation
        extended_obs = np.concatenate([obs, self.z])
        info['z'] = self.z
        return extended_obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        # Concatenate z to observation
        extended_obs = np.concatenate([obs, self.z])
        return extended_obs