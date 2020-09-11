from gym.envs.registration import register
from gridgym.GridEnv import *

register(
	id='grid-v0',
	entry_point='gridgym:GridEnv',
)