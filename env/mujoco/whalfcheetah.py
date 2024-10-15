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
        self.max_y = 2
        self.min_y = 0
        self.frame_skip = 5
        self.model_path = os.path.dirname(os.path.abspath(__file__))
        self.model = load_model_from_path(self.model_path+'/'+self.name+'.xml')
        self.sim = MjSim(self.model)
        self.viewer = None
        self.scale_a = 1
        self.action_space = gym.spaces.Box(low=-self.scale_a, high=self.scale_a, shape=(self.sim.data.ctrl.shape[0],))
        self.s_m = None
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=((self.sim.data.qpos.shape[0]+
                                                                                  self.sim.data.qvel.shape[0] + 
                                                                                  self.sim.data.get_body_xpos("torso").shape[0]),))
        self.render_mode =  render_mode

    def reset(self):
        self.sim.reset()
        self.sim.step()
        s0 = np.concatenate([self.sim.data.qpos.flat.copy(),self.sim.data.qvel.flat.copy(),self.sim.data.get_body_xpos("torso").copy()])
        return s0, {'pos':self.sim.data.get_body_xpos("torso").copy()[:]}


    def step(self, action):
        self.sim.data.ctrl[:] = np.clip(action, -self.scale_a,self.scale_a)/self.scale_a
        for _ in range(self.frame_skip):
            self.sim.step()
        s = np.concatenate([self.sim.data.qpos.flat.copy(),self.sim.data.qvel.flat.copy(),self.sim.data.get_body_xpos("torso").copy()])
        return s, 0, False, False, {'pos':self.sim.data.get_body_xpos("torso").copy()[:]}
    
   

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
    # print(s[-3])
    print(i['pos'])
    # img = env.render()
    # # print(env.action_space)
    # # print(env.gym_equivalent.action_space)
    # # print(env.observation_space)
    # # print(env.gym_equivalent.observation_space)
    # observation = env.reset()
    # for _ in range(500):
    #     action = env.gym_equivalent.action_space.sample() # Ou votre propre logique d'action
    #     observation, reward, done, trunc,  info = env.step(action)
    # #     # if done:
    # #     #     break
    # # # Fermer l'environnement
    # # env.close_r()