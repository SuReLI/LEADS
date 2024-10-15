import gym 

class WHand(gym.Env):
    def __init__(self, name = 'hand',  render=False, render_mode = 'rgb_array', seed = 0):
        super(WHand, self).__init__()
        self.max_steps = 50
        self.name = 'hand'
        self.env_w = gym.make('HandReach-v1', max_episode_steps=self.max_steps) if not render else gym.make('HandReach-v1', max_episode_steps=self.max_steps, render_mode='rgb_array')
        self.action_space = self.env_w.action_space
        self.observation_space = self.env_w.observation_space['observation']
        self.render_mode = 'rgb_array'
        self.t = 0 

    def reset(self):
        self.t = 0
        obs = self.env_w.reset()[0]['observation'].copy()
        return obs, {'pos' : obs[48:]}
    
    def step(self, action):
        self.t += 1
        obs, _, _, _, _ = self.env_w.step(action)
        done = False
        if self.t == self.max_steps:
            done = True
        obs = obs['observation'].copy()
        return obs, 0.0, done, False, {'pos' : obs[48:]}
    
    def render(self):
        camera = self.env_w.render()
        return camera


    
if __name__ == '__main__':
    import numpy as np
    env = WHand(render=True)
    s,i = env.reset()
    img = env.render()
    # print(dir(env.env_w))
    # s, i =env.reset()
    # # s, i =env.reset()
    # # print(env.observation_space)
    # # print(env.action_space)
    # # print(i)
    # for i in range(1000):
    #     a = np.zeros(env.action_space.shape)
    #     obs, _, d, t, i = env.step(a)
    #     print('info: ', i)
    #     env.render()
    # env.close()