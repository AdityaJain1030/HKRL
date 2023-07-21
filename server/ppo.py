import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from copy import deepcopy
# from stable_baselines3 import

from torch.distributions import Categorical

class PPO(nn.Module):
	def __init__(self, obs_size, act_size, gamma = 0.99, clip=0.1, critic_weight=0.5, entropy_weight=0.01, lr_actor=0.0001, lr_critic=0.0001, gae_lambda=0.95):
		super(PPO, self).__init__()
		self.obs_size = obs_size
		self.act_size = act_size

		#define layers
		# self.head = nn.Sequential( # Shared Input layer
		# 	nn.Linear(obs_size, 64),
		# 	nn.ReLU(),
		# 	nn.Linear(64, 64),
		# 	nn.ReLU()
		# )

		self.actor = nn.Sequential(
			nn.Linear(obs_size, 64),
			nn.ReLU(),
			nn.Linear(64, 64),
			nn.ReLU(),
			nn.Linear(64, act_size),
			nn.Softmax(dim=-1)
		)

		self.old_actor = deepcopy(self.actor)

		self.critic = nn.Sequential(
			nn.Linear(obs_size, 64),
			nn.ReLU(),
			nn.Linear(64, 64),
			nn.ReLU(),
			nn.Linear(64, 1)
		)

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
		obs = torch.tensor(obs, dtype=torch.float32)
		dist = self.forward(obs)
		action = dist.sample()
		return action
	
	# OBS SHOULD HAVE 1 MORE STEP THAN REWARDS AND DONES
	def batch_update(self, obs: torch.Tensor, acts: torch.Tensor, rews: np.array, dones: np.array, update_iters = 10):
		advantages = self.calc_batch_advantages(obs, rews, dones) # We can update the critic during the loop now as we already have our fixed advantages

		obs = obs[:, :-1]
		for _ in range(update_iters):
			log_probs = self.forward(obs).log_prob(acts)
			old_log_probs = self.old_forward(obs).log_prob(acts).detach()

			r = torch.exp(log_probs - old_log_probs)

			surrogate_loss = torch.min(r * advantages, torch.clamp(r, 1-self.clip, 1+self.clip) * advantages)
			# print(advantages.shape)
			# print(self.critic(obs).shape)
			critic_loss = F.mse_loss(self.critic(obs).squeeze(dim=-1), advantages)
			entropy_loss = self.forward(obs).entropy()

			loss = -surrogate_loss + (self.critic_weight * critic_loss) - (self.entropy_weight * entropy_loss)

			self.optimizer.zero_grad()
			loss.mean().backward()
			self.optimizer.step()

		kl = (old_log_probs - log_probs).mean().item()

		self.old_actor.load_state_dict(self.actor.state_dict())

		return surrogate_loss.mean().item(), critic_loss.mean().item(), entropy_loss.mean().item(), kl		

	def calc_batch_advantages(self, obs: torch.Tensor, rews: np.array, dones: np.array) -> torch.Tensor:
		# Get critic value estimations
		value_est = self.critic(obs).squeeze(dim=-1).detach().numpy()

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


		advantages = torch.tensor(batch_adv, dtype=torch.float32)
		advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

		return advantages
	
def main():
	import gym
	env = gym.vector.make("CartPole-v1", num_envs=4)
	obs_size = env.observation_space.shape[1]
	act_size = env.action_space[0].n

	agent = PPO(obs_size, act_size)

	obs = env.reset()
	done = False
	
	obs_ = []
	acts_ = []
	rews_ = []
	dones_ = []

	#metrics
	ep_rews = []

	for epoch in range(256):
		for _ in range(4000):
			acts = agent.get_actions(obs)
			next_obs, rew, done, _ = env.step(acts.numpy())

			obs_.append(obs)
			acts_.append(acts.tolist())
			rews_.append(rew)
			ep_rews.append(rew/4)
			dones_.append(done)
			obs = next_obs
		
		obs_.append(obs)
		
		t_acts = np.array(acts_).transpose()
		t_obs = np.array(obs_).transpose(1, 0, 2)
		t_obs = torch.tensor(t_obs, dtype=torch.float32)
		t_acts = torch.tensor(t_acts, dtype=torch.int32)
		t_rews = np.array(rews_).transpose()
		t_dones = np.array(dones_).transpose()

		sl, cl, el, kl = agent.batch_update(t_obs, t_acts, t_rews, t_dones)

		print(f"Epoch: {epoch} | Actor Loss: {sl} | Critic Loss: {cl} | Entropy: {el} | KL Divergence: {kl} | rewards: {np.sum(ep_rews)}")

		if kl > .01:
			break
			# agent.old_actor.load_state_dict(agent.actor.state_dict())
		obs_ = []
		acts_ = []
		rews_ = []
		ep_rews = []
		dones_ = []

	env.close()
	env = gym.make("CartPole-v1")
	obs = env.reset()
	done = False
	while not done:
		action = agent.get_actions(torch.tensor(obs, dtype=torch.float32).unsqueeze(dim=0))
		obs, rew, done, _ = env.step(action.squeeze(dim=0).numpy())
		print(rew)
		env.render()
	env.close()
		

if __name__ == "__main__":
	main()