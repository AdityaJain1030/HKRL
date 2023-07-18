import torch
import torch.nn as nn

from dqn import DQN


class CnnDQN(DQN):
	def __init__(self, obs_size, act_size, use_pooling = False, lr=0.0001, gamma=0.99, replay_buffer_size = 1000000, clip = 1, model_params = None):
		self.channels = obs_size[0]
		self.use_pooling = use_pooling

		super(CnnDQN, self).__init__(obs_size, act_size, lr, gamma, replay_buffer_size, clip, model_params)

	def get_head(self):
		if self.use_pooling:
			return nn.Sequential(
				nn.Conv2d(self.channels, 32, kernel_size=8, stride=4, padding=2),
				nn.ReLU(),
				nn.MaxPool2d(kernel_size=2, stride=2),
				nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
				nn.ReLU(),
				nn.MaxPool2d(kernel_size=2, stride=2),
				nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
				nn.ReLU(),
				nn.MaxPool2d(kernel_size=2, stride=2),
				nn.Flatten(),
			)
		else:
			return nn.Sequential(
				nn.Conv2d(self.channels, 32, kernel_size=8, stride=4, padding=2),
				nn.ReLU(),
				nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
				nn.ReLU(),
				nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
				nn.ReLU(),
				nn.Flatten(),
			)

