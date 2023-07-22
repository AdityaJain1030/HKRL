import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from copy import deepcopy
# from stable_baselines3 import

from torch.distributions import Categorical

import gymnasium as gym

class PPO(nn.Module):
	def __init__(self, obs_size, act_size, gamma=0.99, clip=0.1, critic_weight=0.5, entropy_weight=0.01, lr_actor=0.0001, lr_critic=0.0001, gae_lambda=0.95):
		super(PPO, self).__init__()
		self.obs_size = obs_size
		self.channels = obs_size[0]
		self.act_size = act_size

		self.device = torch.device(
			"cuda:0" if torch.cuda.is_available() else "cpu")

		# define layers
		# self.head = nn.Sequential( # Shared Input layer
		# 	nn.Linear(obs_size, 64),
		# 	nn.ReLU(),
		# 	nn.Linear(64, 64),
		# 	nn.ReLU()
		# )

		self.conv = nn.Sequential(
			nn.Conv2d(self.channels, 32, kernel_size=8, stride=4, padding=2),
			nn.ReLU(),
			nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
			nn.ReLU(),
			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.Flatten(),
		)
		
		with torch.no_grad():
			out_size = self.conv(torch.zeros(1, *obs_size)).numel()

		self.actor = nn.Sequential(
			deepcopy(self.conv),
			nn.Linear(out_size, 256),
			nn.ReLU(),
			nn.Linear(256, 256),
			nn.ReLU(),
			nn.Linear(256, act_size),
			nn.Softmax(dim=-1)
		).to(self.device)

		self.old_actor = deepcopy(self.actor).to(self.device)

		self.critic = nn.Sequential(
			deepcopy(self.conv),
			nn.Linear(out_size, 256),
			nn.ReLU(),
			nn.Linear(256, 256),
			nn.ReLU(),
			nn.Linear(256, 1)
		).to(self.device)

		self.clip = clip
		self.critic_weight = critic_weight
		self.entropy_weight = entropy_weight

		self.lr_actor = lr_actor
		self.lr_critic = lr_critic
		self.gamma = gamma
		self.gae_lambda = gae_lambda

		self.optimizer = torch.optim.Adam([
			{'params': self.actor.parameters(), 'lr': self.lr_actor},
			{'params': self.critic.parameters(), 'lr': self.lr_critic}
		])

	def forward(self, x: torch.Tensor):
		act_dist = Categorical(self.actor(x))
		return act_dist

	def old_forward(self, x: torch.Tensor):
		act_dist = Categorical(self.old_actor(x))
		return act_dist

	def get_actions(self, obs: torch.Tensor):
		dist = self.forward(obs)
		action = dist.sample()
		return action.detach().cpu()

	# OBS SHOULD HAVE 1 MORE STEP THAN REWARDS AND DONES
	def batch_update(self, obs: torch.Tensor, acts: torch.Tensor, rews: np.array, dones: np.array, update_iters=10):
		obs = obs.to(self.device)
		acts = acts.to(self.device)
		# We can update the critic during the loop now as we already have our fixed advantages
		advantages = self.calc_batch_advantages(obs, rews, dones)

		obs = obs[:-1]
		for _ in range(update_iters):
			log_probs = self.forward(obs).log_prob(acts)
			old_log_probs = self.old_forward(obs).log_prob(acts).detach()

			r = torch.exp(log_probs - old_log_probs)

			surrogate_loss = torch.min(
				r * advantages, torch.clamp(r, 1-self.clip, 1+self.clip) * advantages)
			critic_loss = F.mse_loss(self.critic(
				obs).squeeze(dim=-1).unsqueeze(dim=0), advantages)
			entropy_loss = self.forward(obs).entropy()

			loss = -surrogate_loss + \
				(self.critic_weight * critic_loss) - \
				(self.entropy_weight * entropy_loss)

			self.optimizer.zero_grad()
			loss.mean().backward()
			self.optimizer.step()

		kl = (old_log_probs - log_probs).mean().item()

		self.old_actor.load_state_dict(self.actor.state_dict())

		return surrogate_loss.mean().item(), critic_loss.mean().item(), entropy_loss.mean().item(), kl
	
	def batch_act_est(self, batch):
		out = []
		for i in batch:
			dist = self.forward(i).sample()
			out.append(dist)

	#impl so ass I need one of these
	def batch_val_est(self, batch):
		out = []
		for i in batch:
			# print(i.shape)
			val = self.critic(i).squeeze(dim=-1).detach()
			out.append(val)

		return torch.stack(out)
	
	def save(self, path):
		torch.save({
			'actor': self.actor.state_dict(),
			'critic': self.critic.state_dict(),
			'optim': self.optimizer.state_dict()
		}, path)

	def load(self, path):
		data = torch.load(path)
		self.actor.load_state_dict(data['actor'])
		self.old_actor.load_state_dict(data['actor'])
		self.critic.load_state_dict(data['critic'])
		self.optimizer.load_state_dict(data['optim'])

	def calc_batch_advantages(self, obs: torch.Tensor, rews: np.array, dones: np.array) -> torch.Tensor:
		# Get critic value estimations
		
		# value_est = self.batch_val_est(obs).cpu().numpy()
		value_est = self.critic(obs).squeeze(dim=-1).unsqueeze(dim=0).detach().cpu().numpy()

		# calculate batch td errors
		batch_td_errors = np.zeros_like(rews)
		batch_adv = np.zeros_like(rews)

		for i in range(len(rews)):
			batch_td_errors[i][-1] = rews[i][-1] + self.gamma * (1 - dones[i][-1]) * value_est[i][-1]
			for t in range(len(rews[i])-2, -1, -1):
				batch_td_errors[i][t] = rews[i][t] + self.gamma * (1 - dones[i][t]) * batch_td_errors[i][t+1]

		batch_td_errors = batch_td_errors + self.gamma * (1 - dones) * value_est[:, 1:] - value_est[:, :-1]

		# calculate batch advantages
		for i in range(len(rews)):
			batch_adv[i][-1] = batch_td_errors[i][-1]
			for t in range(len(rews[i])-2, -1, -1):
				batch_adv[i][t] = batch_td_errors[i][t] + self.gamma * self.gae_lambda * batch_adv[i][t+1]

		advantages = torch.tensor(
			batch_adv, dtype=torch.float32).to(self.device)
		advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

		return advantages

def make_env(env):
	env = gym.make(env)
	env = gym.wrappers.AtariPreprocessing(env)
	env = gym.wrappers.FrameStack(env, num_stack=4)
	return env

def main():
	env = make_env("PongNoFrameskip-v4")
	agent = PPO(env.observation_space.shape, env.action_space.n)

	obs, _ = env.reset()
	obs = np.array(obs)
	obs = np.expand_dims(obs, axis=0)

	done = False

	obs_ = []
	acts_ = []
	rews_ = []
	dones_ = []

	avg_rew = 0

	#metrics
	ep_len = 0
	ep_rew = []
	ep_rews = []
	ep_lens = []
	# ep_rew = []
	# ep_len = np.ndarray(8)
	# ep_lens = []
	# ep_rews = []

	for epoch in range(1000):
		for _ in range(10000):
			acts = agent.get_actions(torch.from_numpy(obs).float().to(agent.device))
			# print(acts.shape)
			next_obs, rew, done, _, _ = env.step(acts.cpu().item())

			obs_.append(obs)
			acts_.append(acts.tolist())
			rews_.append(rew)
			dones_.append(done)
			obs = np.array(next_obs)
			obs = np.expand_dims(obs, axis=0)

			#metrics
			ep_rew.append(rew)
			ep_len += 1

			if done:
				ep_rews.append(np.sum(ep_rew))
				ep_lens.append(ep_len)
				ep_rew = []
				ep_len = 0



		
		obs_.append(obs)
		t_obs = torch.from_numpy(np.array(obs_)).float().to(agent.device).transpose(0, 1).squeeze(dim=0)
		t_acts = torch.tensor(acts_, dtype=torch.int64).to(agent.device).transpose(0, 1).squeeze(dim=0)
		t_rews = np.expand_dims(np.array(rews_).transpose(), axis=0)
		t_dones = np.expand_dims(np.array(dones_).transpose(), axis=0)

		obs_ = []
		acts_ = []
		rews_ = []
		dones_ = []

		surrogate_loss, critic_loss, entropy_loss, kl = agent.batch_update(t_obs, t_acts, t_rews, t_dones)
		print(f"Epoch: {epoch} | Actor Loss: {surrogate_loss} | Critic Loss: {critic_loss} | Entropy: {entropy_loss} | KL Divergence: {kl} | avg_rewards: {np.mean(ep_rews)}, avg_ep_len: {np.mean(ep_lens)}")

		if epoch % 10 == 0:
			agent.save(f"models/ppo_{epoch}.pth")


if __name__ == "__main__":
	main()
