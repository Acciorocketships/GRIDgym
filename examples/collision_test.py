from gridgym import *
import gym
import numpy as np
import time

grid = np.zeros((8,8)).astype(bool)
env = gym.make('grid-v0', grid=grid, N_AGENTS=3)

startpos = np.array([[1,1],[1,2],[2,1]])

env.reset(startpos)

time.sleep(1)

print("Testing phasing through")

action = np.array([Actions.UP, Actions.DOWN, Actions.WAIT])

pos, _, _, _ = env.step(action)

assert np.all(startpos==pos)

time.sleep(1)

action = np.array([Actions.RIGHT, Actions.WAIT, Actions.LEFT])

pos, _, _, _ = env.step(action)

assert np.all(startpos==pos)

time.sleep(1)

print("Testing wait and move into")

action = np.array([Actions.UP, Actions.WAIT, Actions.WAIT])

pos, _, _, _ = env.step(action)

assert np.all(startpos==pos)

time.sleep(1)

action = np.array([Actions.WAIT, Actions.DOWN, Actions.LEFT])

pos, _, _, _ = env.step(action)

assert np.all(startpos==pos)

time.sleep(1)

print("Testing move into same spot")

action = np.array([Actions.WAIT, Actions.RIGHT, Actions.UP])

pos, _, _, _ = env.step(action)

assert np.all(startpos==pos)

time.sleep(1)

print("Testing collision due to failed movement")

action = np.array([Actions.UP, Actions.WAIT, Actions.LEFT])

pos, _, _, _ = env.step(action)

assert np.all(startpos==pos)

time.sleep(1)

print("Testing move in unison")

action = np.array([Actions.RIGHT, Actions.DOWN, Actions.RIGHT])

pos, _, _, _ = env.step(action)

newpos = np.array([[2,1],[1,1],[3,1]])

assert np.all(newpos==pos)

time.sleep(1)