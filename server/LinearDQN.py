import torch
import torch.nn as nn
import numpy as np

from dqn import DQN


class LinearDQN(DQN):
	def __init__(self, obs_size, act_size, lr=0.0001, gamma=0.99):
		super(LinearDQN, self).__init__(obs_size, act_size, lr, gamma)

	def get_head(self):
		return nn.Linear(np.prod(self.obs_size), 128)

