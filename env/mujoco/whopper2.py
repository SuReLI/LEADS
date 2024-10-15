import gym 
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

class Whopper(gym.Env):
    def __init__(self, name = 'hopper',  render=False, render_mode = 'rgb_array', seed = 0):
        super(Whopper, self).__init__()
        # coordinate to plot x,z
        self.name = name
        self.max_steps = 500
        self.max_x = 3
        self.min_x = -3
        self.max_y = 2
        self.min_y = 0
        self.env_w = gym.make('Hopper-v3', max_episode_steps=self.max_steps) if not render else gym.make('Hopper-v3', max_episode_steps=self.max_steps, render_mode='rgb_array')
        self.set_all_seed(seed)
        self.action_space = self.env_w.action_space
        # observasion space env_w + shape torso + shape foot
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=((self.env_w.observation_space.shape[0] +
                                                                                    self.env_w.unwrapped.sim.data.get_body_xpos("torso").shape[0] +
                                                                                    self.env_w.unwrapped.sim.data.get_body_xpos("foot").shape[0] ),))
        self.t = 0 
        self.render_mode = render_mode
        self.init_position = np.array([0, 0, 0, 0, 0, 0])
        self.init_velocity = np.array([0, 0, 0, 0, 0, 0])

    def reset(self):
        self.t = 0
        s = self.env_w.reset()[0].copy()
        s = np.concatenate([s, self.env_w.unwrapped.sim.data.get_body_xpos("torso").copy(), self.env_w.unwrapped.sim.data.get_body_xpos("foot").copy()])
        return s.copy(), {'pos' : np.concatenate([self.env_w.unwrapped.sim.data.get_body_xpos("torso").copy(), self.env_w.unwrapped.sim.data.get_body_xpos("foot").copy()])}
    
    def step(self, action):
        self.t += 1
        obs, _, _, _, _ = self.env_w.step(action)
        done =False
        if self.t == self.max_steps:
            done = True
        s = np.concatenate([obs.copy(), self.env_w.unwrapped.sim.data.get_body_xpos("torso").copy(), self.env_w.unwrapped.sim.data.get_body_xpos("foot").copy()])
        return s.copy(), 0, done, False, {'pos' : np.concatenate([self.env_w.unwrapped.sim.data.get_body_xpos("torso").copy(), self.env_w.unwrapped.sim.data.get_body_xpos("foot").copy()])}
    
    def render(self):
        camera = self.env_w.render()
        return camera
    def set_all_seed(self, seed = 0):
        # self.env_w.seed(seed)
        np.random.seed(seed)

    
if __name__ == '__main__':
    import numpy as np
    env = Whopper(render=True)
    s, i = env.reset()
    print('s : ', s)
    # print(env.render().shape)
    
    print(env.env_w.unwrapped.sim.data.qpos)
    print(env.env_w.unwrapped.sim.data.qvel)
    # print torso 
    print(env.env_w.unwrapped.sim.data.get_body_xpos("foot").shape)
    # print(s.shape)
    # s,r,d,tr, i = env.step(env.action_space.sample())
    # print(s.shape)
    # print(env.observation_space)
    # print(env.action_space)
    # print(i)
    # for i in range(1000):
    #     a = np.zeros(env.action_space.shape)
    #     obs, _, d, t, i = env.step(a)
    #     print('info: ', i)
    #     env.render()
    # env.close()