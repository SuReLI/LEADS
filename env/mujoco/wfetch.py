import gym 
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

class WFetch(gym.Env):
    def __init__(self, name = 'fetch',  render=False, render_mode = 'rgb_array'):
        super(WFetch, self).__init__()
        # coordinate to plot x,z
        self.name = 'fetch'
        self.max_steps = 200
        self.max_x = 2
        self.max_y = 2
        self.max_z = 2
        self.min_x = 0.0
        self.min_y = 0.0
        self.min_z = 0.0
        self.env_w = gym.make('FetchReach-v2', max_episode_steps=self.max_steps) if not render else gym.make('FetchReach-v2', max_episode_steps=self.max_steps, render_mode='rgb_array')
        self.action_space = self.env_w.action_space
        self.observation_space = self.env_w.observation_space['observation']
        self.t = 0 
        self.render_mode = render_mode

    def reset(self):
        self.t = 0
        return self.env_w.reset()[0]['observation'].copy(), {'pos' : self.env_w.sim.data.get_site_xpos('robot0:grip')[:].astype(np.float32)}
    
    def step(self, action):
        self.t += 1
        obs, _, _, _, _ = self.env_w.step(action)
        done =False
        if self.t == self.max_steps:
            done = True
        return obs['observation'].copy(), 0.0, done, False, {'pos' : self.env_w.sim.data.get_site_xpos('robot0:grip')[:].astype(np.float32)}
    
    def render(self):
        camera = self.env_w.render()
        return camera
    
if __name__ == '__main__':
    import numpy as np
    env = WFetch(render=True)
    s, i =env.reset()
    # print(env.observation_space)
    # print(env.action_space)
    # print(i)
    for i in range(1000):
        a = np.zeros(env.action_space.shape)
        obs, _, d, t, i = env.step(a)
        print('info: ', i)
        env.render()
    env.close()