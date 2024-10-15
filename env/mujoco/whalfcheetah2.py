import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import gym
from mujoco_py import load_model_from_path, MjSim, MjViewer
from record_video import RecordVideo
import copy
import time
class Whalfcheetah(gym.Env):
    def __init__(self, name='halfcheetah', render = False, render_mode = 'rgb_array'):
        super(Whalfcheetah, self).__init__()
        self.name = 'half_cheetah'
        self.max_steps = 500
        self.max_x = 10
        self.min_x = -10
        self.max_y = 1
        self.min_y = -1
        self.base_env = gym.make('HalfCheetah-v3', render_mode = render_mode, reset_noise_scale=0.0, exclude_current_positions_from_observation =False, max_episode_steps=1000)
        self.render_mode =  render_mode
        self.observation_space = self.base_env.observation_space
        self.action_space = self.base_env.action_space


    def reset(self):
        obs, i = self.base_env.reset()
        return obs, {'pos': obs[:2]}


    def step(self, action):
        obs, reward, done, trunc, info = self.base_env.step(action)
        return obs, reward, done, trunc, {'pos': obs[:2]}
   

    def render(self, mode='rgb_array'):
        if mode == 'rgb_array':
            # Récupérer la largeur et la hauteur de la fenêtre du viewer
            width, height = 800, 600  # Vous pouvez ajuster ces valeurs selon vos besoins

            # Créer un buffer pour l'image
            camera = np.flipud(self.sim.render(width, height, camera_name='track'))  # Utilisez le nom de votre caméra ici

            return camera
        else:
            if self.viewer is None:
                self.viewer = MjViewer(self.sim)
            self.viewer.render()


    def close_r(self) : 
        self.gym_equivalent.close_video_recorder()

if __name__ == '__main__':
    env = Whalfcheetah(render=True)
    s, i = env.reset()
    print(s.shape)
    for i in range(1000):
        a = env.action_space.sample()
        s, r, d, i = env.step(a)
        print(s.shape, r, d, i['pos'])
        # env.render()
        time.sleep(0.01)