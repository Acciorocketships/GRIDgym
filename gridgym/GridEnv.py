import gym
import numpy as np
from enum import IntEnum
import types

class GridEnv(gym.Env):

	def __init__(self, grid, **kwargs):
		# Constants
		self.grid_fn = None; self.grid = None
		if isinstance(grid, types.FunctionType):
			self.grid_fn = grid
		else:
			self.grid = grid
		self.state_fn = lambda env: env.positions
		self.reward_fn = lambda env: 0.0
		self.done_fn = lambda env: False
		self.info_fn = lambda env: {}
		self.N_AGENTS = 1
		self.COMM_RANGE = float('inf')
		self.HEADLESS = False
		self.set_constants(kwargs)
		# Variables
		self.positions = None
		self.collisions = np.zeros(self.N_AGENTS)
		self.goals = {}
		self.steps_since_reset = 0
		self.movement = None
		self.obs = None
		self.random_reset()
		if not self.HEADLESS:
			from gridgym.Visualiser import Visualiser
			self.visualiser = Visualiser()


	def set_constants(self, kwargs):
		for name, val in kwargs.items():
			if name in self.__dict__:
				self.__dict__[name] = val


	def random_reset(self):
		if self.grid_fn is not None:
			self.grid = self.grid_fn()
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


	def get_local_map(self, agent, size=2):
		pos = self.positions[agent,:]
		localmap = np.zeros((2*size+1, 2*size+1, 3))
		# obstacles
		padded_grid = np.ones((self.grid.shape[0]+2*size, self.grid.shape[1]+2*size))
		padded_grid[size:-size,size:-size] = self.grid
		localmap[:,:,0] = padded_grid[size+pos[0]-size:size+pos[0]+size+1, size+pos[1]-size:size+pos[1]+size+1]
		# agents
		local_positions = self.positions - pos[None,:] + size
		for agent_idx, agent_pos in enumerate(local_positions):
			if np.all(agent_pos >= 0) and np.all(agent_pos <= size*2) and agent_idx != agent:
				localmap[agent_pos[0], agent_pos[1], 1] = True
		# goal
		local_goal = self.goals[agent,:] - pos
		scale = max(abs(local_goal))
		local_goal = np.round(local_goal * (size / max(size, scale))).astype(int)
		localmap[local_goal[0]+size, local_goal[1]+size, 2] = True
		# 3 channel map
		return localmap


	# pos: Nx2 array where the kth row is the (i,j) coordinate of agent k
	def reset(self, pos, grid=None):
		if grid is not None:
			self.grid = grid
		if not np.any(self.check_collisions(pos)):
			self.positions = pos
			self.steps_since_reset = 0


	def set_goals(self, goals):
		self.goals = goals


	def get_A(self):
		if self.COMM_RANGE == float('inf'):
			return np.ones((self.N_AGENTS, self.N_AGENTS)) - np.eye(self.N_AGENTS)
		posi = self.positions.unsqueeze(1).expand(-1,self.N_AGENTS,-1)
		posj = self.positions.unsqueeze(0).expand(self.N_AGENTS,-1,-1)
		copos = posi-posj
		codist = copos.norm(dim=2)
		codist.diagonal().fill_(float('inf'))
		A = (codist <= self.COMM_RANGE).float()
		return A


	def get_X(self):
		return self.state_fn(self)


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


action_categorical = {
	(0,0) : Actions.WAIT,
	(0,1) : Actions.UP,
	(0,-1): Actions.DOWN,
	(1,0) : Actions.RIGHT,
	(-1,0): Actions.LEFT
}


def grid_creator(size, fill=0.1):
	if type(size)==int:
		size = (size, size)
	grid = np.random.rand(*size) < fill
	return grid
