import gym
import numpy as np
from enum import IntEnum
from gridgym.Visualiser import *

class GridEnv(gym.Env):

	def __init__(self, grid, **kwargs):
		# Constants
		self.grid = grid
		self.state_fn = lambda env: env.positions
		self.reward_fn = lambda env: 0.0
		self.done_fn = lambda env: False
		self.info_fn = lambda env: {}
		self.N_AGENTS = 1
		self.HEADLESS = False
		self.set_constants(kwargs)
		# Variables
		self.positions = None; self.random_reset()
		self.collisions = np.zeros(self.N_AGENTS)
		self.goals = {}
		self.steps_since_reset = 0
		self.movement = None
		self.obs = None
		if not self.HEADLESS:
			self.visualiser = Visualiser(grid.shape)


	def set_constants(self, kwargs):
		for name, val in kwargs.items():
			if name in self.__dict__:
				self.__dict__[name] = val


	def random_reset(self):
		pos = self.get_valid_positions()
		self.positions = pos
		self.steps_since_reset = 0


	def get_valid_positions(self, N=None):
		if N is None:
			N = self.N_AGENTS
		valid_i, valid_j = np.where(~self.grid)
		idxs = np.random.choice(len(valid_i), N, replace=False)
		pos_i = valid_i[idxs]
		pos_j = valid_j[idxs]
		pos = np.stack((pos_i, pos_j), axis=1)
		return pos

	# pos: Nx2 array where the kth row is the (i,j) coordinate of agent k
	def reset(self, pos):
		if not np.any(self.check_collisions(pos)):
			self.positions = pos
			self.steps_since_reset = 0


	def set_goals(self, goals):
		self.goals = goals


	def check_collisions(self, pos, old_pos=None):
		collisions = np.zeros(pos.shape[0]).astype(bool)
		# Check that agents don't occupy the same space
		pos_dict = {}
		for i in range(pos.shape[0]):
			key = tuple(pos[i,:])
			if key in pos_dict:
				if len(pos_dict[key]) == 1:
					collisions[pos_dict[key]] = True
				collisions[i] = True
				pos_dict[key] += [i]
			else:
				pos_dict[key] = [i]
		if old_pos is not None:
			for i in range(pos.shape[0]):
				if tuple(old_pos[i,:]) in pos_dict:
					for j in pos_dict[tuple(old_pos[i,:])]:
						if i != j:
							if tuple(pos[i,:]) == tuple(old_pos[j,:]):
								collisions[i] = True
								collisions[j] = True
		# Check collisions with edges
		collisions = np.logical_or(collisions, pos[:,0] < 0)
		collisions = np.logical_or(collisions, pos[:,1] < 0)
		collisions = np.logical_or(collisions, pos[:,0] >= self.grid.shape[0])
		collisions = np.logical_or(collisions, pos[:,1] >= self.grid.shape[1])
		# Check collisions with obstacles
		mask = ~collisions
		collisions[mask] = np.logical_or(collisions[mask], self.grid[pos[mask,0],pos[mask,1]])
		# Combination of all collisions
		return collisions


	def step(self, action):
		# Construct movement matrix
		if len(action.shape) == 1:
			movement = np.zeros((self.N_AGENTS, 2)).astype(int)
			movement[action==Actions.UP,1] = 1
			movement[action==Actions.DOWN,1] = -1
			movement[action==Actions.RIGHT,0] = 1
			movement[action==Actions.LEFT,0] = -1
		else:
			movement = action.astype(int)
		# Stop movement if it results in a collision
		collisions = self.check_collisions(self.positions+movement, self.positions)
		movement[collisions,:] = 0
		self.collisions = self.check_collisions(self.positions+movement, self.positions) # second pass with updated movement
		movement[self.collisions,:] = 0
		self.movement = movement
		# Execute action
		self.positions += movement
		# Visualise
		if not self.HEADLESS:
			self.visualiser.render(grid=self.grid, positions=self.positions, goals=self.goals)
		# Update
		self.steps_since_reset += 1
		# Compute data and return
		self.obs = self.state_fn(self)
		reward = self.reward_fn(self)
		done = self.done_fn(self)
		info = self.info_fn(self)
		return self.obs, reward, done, info




class Actions(IntEnum):
	WAIT = 0
	UP = 1
	RIGHT = 2
	DOWN = 3
	LEFT = 4


def grid_creator(size, fill=0.1):
	if type(size)==int:
		size = (size, size)
	grid = np.random.rand(*size) < fill
	return grid
