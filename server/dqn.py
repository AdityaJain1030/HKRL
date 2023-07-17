import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from abc import ABC, abstractmethod

import numpy as np
import copy
from collections import deque
import random

class DQN(nn.Module, ABC):
	def __init__(self, obs_size, act_size, lr=0.0001, gamma=0.99, replay_buffer_size = 1000000, model_params = None):
		super(DQN, self).__init__()

		self.obs_size = obs_size
		self.act_size = act_size
		self.memory = deque(maxlen=replay_buffer_size)

		self.gamma = gamma
		self.lr = lr

		# self.input_layer = 
		with torch.no_grad():
			out_size = self.input_layer(torch.zeros(1, *obs_size)).numel()
		
		self.actions_head = nn.Sequential(
			nn.Linear(out_size, 128),
			nn.ReLU(),
			nn.Linear(128, 128),
			nn.ReLU(),
			nn.Linear(128, act_size)
		)
		self.Q = nn.Sequential(
			self.input_layer,
			self.actions_head
		)

		if model_params is not None:
			self.load_state_dict(model_params)

		self.target = copy.deepcopy(self.Q)

		self.optimizer = optim.Adam(self.Q.parameters(), lr=lr)

	
	@property
	@abstractmethod
	def input_layer(self) -> nn.Module:
		pass

	def forward(self, x):
		return self.Q(x)
	
	def get_actions(self, obs, eplison=0):
		if random.random() > eplison:
			obs = torch.from_numpy(obs).float()

			act_vals = self.Q(obs).unsqueeze(0)
			action = act_vals.max(dim=1)[1].detach().numpy()

			return action
		else:
			return [random.randint(0, self.act_size[0] - 1) for _ in range(obs.shape[0])]
		
	
	def synchronize_target(self, polyak=None):
		if polyak is not None:
			for target_param, param in zip(self.target.parameters(), self.Q.parameters()):
				target_param.data.copy_(polyak * param.data + (1 - polyak) * target_param.data)
		else:
			self.target.load_state_dict(self.Q.state_dict())

	def save(self, path):
		torch.save(self.state_dict(), path)

	def load(self, path):
		self.load_state_dict(torch.load(path))

	def save_replay(self, obs, action, reward, next_obs, done):
		self.memory.append((obs, action, reward, next_obs, done))

	def sample_replay(self, batch_size):
		batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
		obs, act, rew, obs_next, done = map(np.stack, zip(*batch))
		return obs, act, rew, obs_next, done
	
	def train(self, obs, action, reward, next_obs, done):

		obs = torch.from_numpy(obs).float()
		next_obs = torch.from_numpy(next_obs).float()
		action = torch.tensor(action).long()
		reward = torch.tensor(reward).long()
		done = torch.tensor(done).long()


		with torch.no_grad():
			target = reward + self.gamma * self.target(next_obs).max(dim=1)[0] * (1 - done)

		pred = self.Q(obs).gather(1, action.unsqueeze(1)).squeeze(1)
		loss = F.mse_loss(pred, target)
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		return loss.item()
