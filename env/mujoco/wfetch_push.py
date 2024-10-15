import gym 
import numpy as np
import warnings
import mujoco
from gym_robotics.utils import mujoco_utils
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

class WFetch_Push(gym.Env):
    def __init__(self, name = 'fetch_push',  render=False, render_mode = 'rgb_array'):
        super(WFetch_Push, self).__init__()
        # coordinate to plot x,z
        self.name = 'fetch_push'
        self.max_steps = 50
        # gripper
        self.max_x = 2
        self.max_y = 2
        self.max_z = 2
        self.min_x = 0.0
        self.min_y = 0.0
        self.min_z = 0.0
        # puck
        self.max_puck_x = 2
        self.max_puck_y = 2
        self.min_puck_x = 0.0
        self.min_puck_y = 0.0
        self.env_w = gym.make('FetchSlide-v2', max_episode_steps=self.max_steps) if not render else gym.make('FetchSlide-v2', max_episode_steps=self.max_steps, render_mode=render_mode)
        self.action_space = self.env_w.action_space
        self.observation_space = self.env_w.observation_space['observation']
        self.t = 0 
        self.render_mode = render_mode
        self.env_w.obj_range = 0.0
        self.env_w.target_range = 0.0
        self.env_w.target_in_the_air = False
        self.env_w.has_object = True
        self.env_w.reset()
        # initial
        self.env_w.initial_gripper_xpos = np.array([1.03, 0.74904077,0.41266129])
        # print('initial_gripper_xpos : ', self.env_w.initial_gripper_xpos)
        # change z 
        # self.env_w.initial_qpos[2] = 0.4
        # self.env_w.target_offset = 0.0

    def reset(self):
        #set the gripper position
        self.env_w.data.time = self.env_w.initial_time
        self.env_w.data.qpos[:] = np.copy(self.env_w.initial_qpos)
        self.env_w.data.qvel[:] = np.copy(self.env_w.initial_qvel)
        if self.env_w.model.na != 0:
            self.env_w.data.act[:] = None
        # set the puck position
        object_xpos = self.env_w.initial_gripper_xpos[:2]
        object_qpos = mujoco_utils.get_joint_qpos(
                self.env_w.model, self.env_w.data, "object0:joint"
            )
        object_qpos[:2] = object_xpos
        mujoco_utils.set_joint_qpos(
            self.env_w.model, self.env_w.data, "object0:joint", object_qpos
        )
        mujoco.mj_forward(self.env_w.model, self.env_w.data)
        # get observation
        (grip_pos,
        object_pos,
        object_rel_pos,
        gripper_state,
        object_rot,
        object_velp,
        object_velr,
        grip_velp,
        gripper_vel,
        ) = self.env_w.generate_mujoco_observations()
        o = np.concatenate(
            [grip_pos,
            object_pos.ravel(),
            object_rel_pos.ravel(),
            gripper_state,
            object_rot.ravel(),
            object_velp.ravel(),
            object_velr.ravel(),
            grip_velp,
            gripper_vel,
            ]
        ).copy()
        return o.astype(np.float32), {'pos' : o[:6].astype(np.float32)}
    
    def step(self, action):
        self.t += 1
        obs, _, _, _, _ = self.env_w.step(action)
        done =False
        if self.t == self.max_steps:
            done = True
        o=obs['observation'].copy()
        return o, 0.0, done, False, {'pos' : o[:6].astype(np.float32)}
    
    def render(self):
        img = self.env_w.render()
        return img

    
if __name__ == '__main__':
    import numpy as np
    env = WFetch_Push(render=False)
    s1, i =env.reset()
    print('s : ', s1)
    s, i =env.reset()
    print('s : ', s)
    print(s1 == s)
    # print(env.env_w.initial_gripper_xpos)
    # print(env.env_w.initial_object_xpos)
    # print(np.concatenate(env.env_w.generate_mujoco_observations()).shape)
    # # print(env.observation_space)
    # # print(env.action_space)
    # # print(i)
    # for i in range(1000):
    #     a = np.zeros(env.action_space.shape)
    #     obs, _, d, t, i = env.step(a)
    #     print('info: ', i)
    #     # env.render()
    # env.close()