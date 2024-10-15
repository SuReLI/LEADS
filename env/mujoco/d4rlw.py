# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility for loading the AntMaze environments."""
import d4rl
import gym
import numpy as np


R = 'r'
G = 'g'
U_MAZE = [[1, 1, 1, 1, 1],
          [1, 0, 0, 0, 1],
          [1, 0, R, 0, 1],
          [1, 0, 0, 0, 1],
          [1, 1, 1, 1, 1]]



class D4rlw(d4rl.locomotion.ant.AntMazeEnv):
  """Utility wrapper for the AntMaze environments.

  For comparisons in the offline RL setting, we used unmodified AntMaze tasks,
  without this wrapper.
  """

  def __init__(self, map_name, non_zero_reset=False):
    self._goal_obs = np.zeros(29)
    if map_name == 'umaze':
      maze_map = U_MAZE
    else:
      raise NotImplementedError
    super(D4rlw, self).__init__(maze_map=maze_map,
                                  reward_type='sparse',
                                  non_zero_reset=non_zero_reset,
                                  eval=True,
                                  maze_size_scaling=4.0,
                                  ref_min_score=0.0,
                                  ref_max_score=1.0)
    self.observation_space = gym.spaces.Box(
        low=np.full((58,), -np.inf),
        high=np.full((58,), np.inf),
        dtype=np.float32)

  def reset(self):
    super(D4rlw, self).reset()
    goal_xy = (0,0)
    state = self.sim.get_state()
    state = state._replace(
        qpos=np.concatenate([goal_xy, state.qpos[2:]]))
    self.sim.set_state(state)
    for _ in range(50):
      self.do_simulation(np.zeros(8), self.frame_skip)
    self._goal_obs = self.BASE_ENV._get_obs(self).copy()  # pylint: disable=protected-access
    super(D4rlw, self).reset()
    qpos = self.init_qpos.copy()
    qvel = self.init_qvel.copy()
    self.set_state(qpos, qvel)
    return self._get_obs()

  def step(self, action):
    super(D4rlw, self).step(action)
    s = self._get_obs()
    dist = np.linalg.norm(self._goal_obs[:2] - s[:2])
    # Distance threshold from [RIS, Chane-Sane '21] and [UPN, Srinivas '18].
    r = (dist <= 0.5)
    done = False
    info = {}
    return s, r, done, info

  def _get_obs(self):
    assert self._expose_all_qpos  # pylint: disable=protected-access
    s = self.BASE_ENV._get_obs(self)  # pylint: disable=protected-access
    return np.concatenate([s, self._goal_obs]).astype(np.float32)

  def _get_reset_location(self):
    if np.random.random() < 0.5:
      return super(D4rlw, self)._get_reset_location()
    else:
      return self._goal_sampler(np.random)
    

if __name__ == '__main__':
    import time 
    env = D4rlw('umaze')
    print(env._expose_body_coms)
    print(env._expose_body_comvels)
    print(env.physics.data.qpos.flat[:15].shape,  # Ensures only ant obs.
          env.physics.data.qvel.flat[:14].shape)
    print(env._get_obs().shape)
    # print(env.observation_space)
    # print(env.action_space)
    # print(env.action_space)
    # env.unwrapped.frame_skip = 100
    # s = env.reset()
    # print(s)
    # print( 'shape : ', s.shape  )
    # s,r,d,i = env.step(env.action_space.sample())
    # print(s.shape)
    # # print('s0 : ', s)
    # for k in range(1000):
    #     action = env.action_space.sample()
    #     obs, reward, done, concatenate, info = env.step(action)
    #     # print(env.sim.data.get_body_xpos('torso')[:2])

    #     time.sleep(0.1)

    #     # print(info)
    #     env.render()
    # #     if done:
    # #         env.reset()