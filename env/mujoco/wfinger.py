from dm_control import suite 
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.suite.utils import randomizers
from dm_control.utils import containers
from gym.spaces import Box
import matplotlib.pyplot as plt
import gym
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

class WFinger(gym.Env):
	def __init__(self,name='finger', max_steps=50, render = False, render_mode = 'rgb_array'):
		super(WFinger, self).__init__()
		self.env=suite.load("finger","turn_hard",task_kwargs={"time_limit":float("inf")})
		self.observation_space_dim=9
		self.action_space_dim=2
		self.action_space=Box(low=-np.ones(self.action_space_dim), high=np.ones(self.action_space_dim))
		self.observation_space=Box(low=-1*np.ones(self.observation_space_dim), high=np.ones(self.observation_space_dim))
		self.time=0
		self.type='dm_control'
		self.name='finger'
		self.max_steps = max_steps
		# theta 
		self.min_x=-np.pi
		self.max_x=np.pi
		self.min_y=-np.pi
		self.max_y=np.pi
		self.min_z=-np.pi
		self.max_z=np.pi
		# render
		self.render_mode = render_mode
		if render and render_mode == 'human': 
			self.fig, self.ax = plt.subplots()
			self.img = None
			self.initialized = False

	def reset(self):
		# self.env.reset()
		# qpos 
		self.env._physics.named.data.qpos["proximal"]=np.pi/2
		self.env._physics.named.data.qpos["distal"]=0
		self.env._physics.named.data.qpos["hinge"]=np.pi/2
		# site_pose
		self.env._physics.model.site_pos=np.array([[ 0.3168727, 0., 0.45692778]
													,[ 0.01, 0. , -0.17 ]
													,[-0.01, 0. , -0.17 ]
													,[ 0.  , 0.  , 0.13 ]])
		# site_size
		self.env._physics.model.site_size=np.array([[0.03,  0.005, 0.005]
													,[0.025, 0.03,  0.025]
													,[0.025, 0.03,  0.025]
													,[0.02,  0.005, 0.005]])
		self.env._task.before_step([0,0], self.env._physics)
		self.env._physics.step(self.env._n_sub_steps)
		self.env._task.after_step(self.env._physics)
		state=self.env._task.get_observation(self.env._physics)
		position,velocity,touch=state["position"],state["velocity"],state["touch"]
		state=np.concatenate((position,velocity,touch),axis=0)
		self.time=0
		return state, {'pos':list(position)}
	
	def step(self,action):
		self.time += 1
		self.env._task.before_step(action, self.env._physics)
		self.env._physics.step(self.env._n_sub_steps)
		self.env._task.after_step(self.env._physics)
		state=self.env._task.get_observation(self.env._physics)
		reward=self.env._task.get_reward(self.env._physics)
		# transorfmation
		position,velocity,touch=state["position"],state["velocity"],state["touch"]
		state=np.concatenate((position,velocity,touch),axis=0)
		done= False
		if self.time>=self.max_steps:
			done=True
		reward=0.0
		return state, reward, done, False, {'pos':list(position)}

	def render(self):
		return self.env.physics.render()
	
	def render(self, mode='rgb_array'):
		image = self.env.physics.render()
		if mode == 'human':
			if not self.initialized:
				self.img = self.ax.imshow(image)
				plt.show(block=False)
				self.initialized = True
			else:
				self.img.set_data(image)
				self.fig.canvas.draw()
				self.fig.canvas.flush_events()
		else:
			return image
		

if __name__=='__main__' : 
	import matplotlib.pyplot as plt 
	env = WFinger(render=True)
	# print(env.env.action_spec())
	# print(env.env.observation_spec())
	s,i = env.reset()
	print('i : ', i)
	# # print(s.shape)
	# for k in range(1000) : 
	# 	a = np.ones(env.action_space_dim)
	# 	s,r,d,tr,i = env.step(a)
	# 	env.render()
	# 	print('state : ', s)

	# # env.step(env.action_space.sample())
	# # print(s)
	