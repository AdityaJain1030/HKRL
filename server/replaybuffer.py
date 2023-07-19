from collections import deque
import random
import numpy as np

class ReplayBuffer:
	def __init__(self, size):
		self.size = size
		self.buffer = deque(maxlen=size)
	
	def push(self, obs, action, reward, next_obs, done):
		self.buffer.append((obs, action, reward, next_obs, done))

	def push_batch(self, obs, action, reward, next_obs, done):
		for i in range(len(obs)):
			self.push(obs[i], action[i], reward[i], next_obs[i], done[i])

	def sample(self, batch_size):
		batch = random.sample(list(enumerate(self.buffer)), min(batch_size, self.size))
		index, batch = zip(*batch)
		obs, act, rew, obs_next, done = map(np.stack, zip(*batch))
		return obs, act, rew, obs_next, done, np.array(index)