import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical

import gymnasium as gym
import numpy as np


class VPG(nn.Module):
	def __init__(self, obs_size, act_size, lr=0.001, gamma=.99):
		super(VPG, self).__init__()

		self.gamma = gamma
		self.lr = lr

		self.actor = nn.Sequential(
			nn.Linear(obs_size, 128),
			nn.ReLU(),
			nn.Linear(128, act_size),
		)

		self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

	def forward(self, x):
		return self.actor(x)

	def get_action(self, obs):
		obs = torch.tensor(obs, dtype=torch.float32)
		action = Categorical(logits=self.actor(obs)).sample().item()
		return action

	def _get_logprob(self, obs, act):
		return Categorical(logits=self.actor(obs)).log_prob(act)

	def train(self, obs, act, rew, dones):
		obs = torch.tensor(obs, dtype=torch.float32)
		act = torch.tensor(act, dtype=torch.int64)
		rew = self._get_rtgs(rew, dones)

		logprob = self._get_logprob(obs, act)
		loss = -(logprob * rew).mean()

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		return loss.item()

	def save(self, path):
		torch.save(self.state_dict(), path)

	def load(self, path):
		self.load_state_dict(torch.load(path))

	def _get_rtgs(self, rews, dones):
		result = np.empty_like(rews)
		result[-1] = rews[-1]
		for t in range(len(rews)-2, -1, -1):
			result[t] = rews[t] + self.gamma*(1-dones[t])*result[t+1]
		return torch.tensor(result).float()

	
def main():
	env = gym.make('CartPole-v1')
	agent = VPG(env.observation_space.shape[0], env.action_space.n)

	for i in range(16 * 200):
		done = False
		truncated = False
		ep_rew = []
		obs, _ = env.reset()
		ep_len = 0
		traj = []
		for j in range(1000):
			act = agent.get_action(obs)
			next_obs, rew, done, _, _ = env.step(act)
			ep_rew.append(rew)
			ep_len += 1
			traj.append((obs, act, rew, done))
			obs = next_obs
			
			if done:
				done = False
				truncated = False
				ep_rew = []
				obs, _ = env.reset()
				ep_len = 0

				

		loss = agent.train(*zip(*traj))

		print(f'Episode {i} | Episode Reward: {sum(ep_rew)} | Episode Length: {ep_len} | Loss: {loss}')

	env.close()

	agent.save('vpg.pt')


def eval():
	env = gym.make('CartPole-v1', render_mode='human')
	agent = VPG(env.observation_space.shape[0], env.action_space.n)
	agent.load('vpg.pt')
	obs, _ = env.reset()
	done = False
	while not done:
		act = agent.get_action(obs)
		obs, _, done, _, _ = env.step(act)

main()
eval()
