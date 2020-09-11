from gridgym import *
import gym
import numpy as np
import time

grid = np.random.rand(8,8) < 0.25
env = gym.make('grid-v0', grid=grid, N_AGENTS=3)

goals = env.get_valid_positions()
env.set_goals(goals)

while True:
	# print(env.positions)
	# print(env.collisions)
	action = np.random.randint(5, size=env.N_AGENTS)
	env.step(action)
	time.sleep(0.2)