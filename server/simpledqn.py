import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import copy
from collections import deque
import gymnasium as gym

import tensorflow as tf
import tensorboard as tb
from torch.utils.tensorboard import SummaryWriter

tf.io.gfile = tb.compat.tensorflow_stub.io.gfile



class DQN(nn.Module):
	def __init__(self, obs_size, act_size, lr=0.0001, gamma=0.99):
		super(DQN, self).__init__()
		self.obs_size = (obs_size,)
		self.act_size = (act_size,)
		
		self.Q = nn.Sequential(
			 nn.Linear(obs_size, 128),
			 nn.ReLU(),
			 nn.Linear(128, 128),
			 nn.ReLU(),
			 nn.Linear(128, act_size)
		)
		
		self.target = copy.deepcopy(self.Q)
		self.gamma = gamma

		self.optimizer = optim.Adam(self.Q.parameters(), lr=lr)
	
	def forward(self, x):
		return self.Q(x)
	
	def synchronize_target(self):
		self.target = copy.deepcopy(self.Q)

	def get_action(self, obs, eplison):
		if random.random() > eplison:
			obs = torch.from_numpy(obs).float()

			act_vals = self.Q(obs).unsqueeze(0)
			action = act_vals.max(dim=1)[1].detach().numpy()
			if len(action) == 1:
				return action[0]

			return action
			
		else:
			if len(obs.shape) == len(self.obs_size):
				return random.randint(0, self.act_size[0] - 1)
			else:
				return [random.randint(0, self.act_size[0] - 1) for _ in range(obs.shape[0])]
	
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

class ReplayBuffer:
	def __init__(self, size):
		self.size = size
		self.buffer = deque(maxlen=size)
	
	def push(self, obs, action, reward, next_obs, done):
		self.buffer.append((obs, action, reward, next_obs, done))

	def sample(self, batch_size):
		batch = random.sample(self.buffer, min(batch_size, self.size))
		obs, act, rew, obs_next, done = map(np.stack, zip(*batch))
		return obs, act, rew, obs_next, done

def main():
	# import pytorch tensorboard
	writer = SummaryWriter()
	env = gym.make('CartPole-v1')
	obs_size = env.observation_space.shape[0]
	act_size = env.action_space.n

	agent = DQN(obs_size, act_size, lr=0.0001, gamma=0.99)
	buffer = ReplayBuffer(100000)

	eplison = init_eplison = .95
	e_greedy_steps = 1000
	final_epsilon = 0.01
	tt = 0
	episodes = 250
	max_timesteps = 200
	for e in range(episodes):
		obs, _ = env.reset()
		ep_rews = []
		ep_len = 0
		ep_loss = []
		for t in range(max_timesteps):
			if t % 10 == 0:
				agent.synchronize_target()
						
			action = agent.get_action(obs, eplison)
			next_obs, reward, done, _, _ = env.step(action)
			ep_len += 1
			tt += 1
			ep_rews.append(reward)
			buffer.push(obs, action, reward, next_obs, done)
			obs = next_obs

			eplison = max(final_epsilon, init_eplison - (init_eplison - final_epsilon) * tt / e_greedy_steps)

			if tt > 100:
				s, a, r, s_n, d = buffer.sample(32)
				loss = agent.train(s, a, r, s_n, d)
				ep_loss.append(loss)

			if done:
				break
			
		print(f"Episode: {e}, ep_len: {ep_len}, avg_loss: {np.mean(ep_loss)}, reward: {np.sum(ep_rews)}")
		writer.add_scalar("loss", np.mean(ep_loss), e)
		writer.add_scalar("reward", np.sum(ep_rews), e)
		writer.add_scalar("ep_len", ep_len, e)
		writer.add_scalar("eplison", eplison, e)


main()