import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.distributions import Categorical

import numpy as np
import gymnasium as gym
from copy import deepcopy
from logger import Logger
from torchviz import make_dot

from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack


class ActorCritic(nn.Module):
	def __init__(self, obs_size, act_size, share_extractor=False):
		super(ActorCritic, self).__init__()

		self.obs_size = obs_size
		self.channels = obs_size[0]
		self.act_size = act_size
		self.share_extractor = share_extractor

	# 	NOTE: Make sure to normalize conv input
		# self.extractor = nn.Sequential(
		# 	nn.Conv2d(self.channels, 32, kernel_size=8, stride=4),
		# 	nn.ReLU(),
		# 	nn.Conv2d(32, 64, kernel_size=4, stride=2),
		# 	nn.ReLU(),
		# 	nn.Conv2d(64, 64, kernel_size=3, stride=1),
		# 	nn.ReLU(),
		# 	nn.Flatten()
		# )

		self.extractor = nn.Sequential(
			nn.Linear(np.prod(obs_size), 64),
			nn.ReLU()
		)

		if not self.share_extractor:
			self.extractor_copy = deepcopy(self.extractor)

		with torch.no_grad():
			out_size = self.extractor(
				torch.zeros(1, *obs_size)).detach().numel()
			print("out_size: ", out_size)

		self.actor = nn.Sequential(
			nn.Linear(out_size, 256),
			nn.ReLU(),
			nn.Linear(256, act_size),
			nn.Softmax(dim=-1)
		)

		self.critic = nn.Sequential(
			nn.Linear(out_size, 256),
			nn.ReLU(),
			nn.Linear(256, 1)
		)

	def get_value(self, obs):
		if self.share_extractor:
			x = self.extractor(obs)
		else:
			x = self.extractor_copy(obs)
		return self.critic(x)

	def get_action(self, obs):
		act_probs = self.actor(self.extractor(obs))
		dist = Categorical(act_probs)
		action = dist.sample()
		return action, dist.log_prob(action), dist.entropy()

	def get_log_probs(self, obs, acts):
		act_probs = self.actor(self.extractor(obs))
		dist = Categorical(act_probs)
		return dist.log_prob(acts)


class PPO:
	def __init__(self, obs_size, act_size, lr=0.0001, gamma=0.99, clip=0.2, value_coeff=0.5, entropy_coeff=0.01,
				 max_grad_norm=0.5, lam=0.95, target_kl=None, share_extractor=True, model_params=None, anneal_lr=None):
		self.obs_size = obs_size
		self.act_size = act_size
		self.gamma = gamma
		self.learning_rate = lr
		self.lam = lam
		self.max_grad_norm = max_grad_norm
		self.share_extractor = share_extractor
		self.anneal_lr = False if anneal_lr is None else True

		self.target_kl = target_kl
		self.clip = clip
		self.value_coeff = value_coeff
		self.entropy_coeff = entropy_coeff

		self.device = torch.device(
			"cuda" if torch.cuda.is_available() else "cpu")

		self.policy = ActorCritic(
			self.obs_size, self.act_size, share_extractor).to(self.device)
		self.policy.apply(self.init_orthog_weights)

		if model_params is not None:
			self.policy.load_state_dict(model_params)

		self.optimizer = optim.Adam(
			self.policy.parameters(), lr=self.learning_rate)

		if self.anneal_lr:
			self.scheduler = torch.optim.lr_scheduler.LambdaLR(
				self.optimizer, lr_lambda=lambda epoch: 1 - epoch / anneal_lr)

	def init_orthog_weights(self, m):
		if type(m) == nn.Linear:
			nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
			nn.init.zeros_(m.bias)
		if type(m) == nn.Conv2d:
			nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
			nn.init.zeros_(m.bias)

	# Taken from cleanRL implementation
	def get_advantages(self, rewards, dones, values):
		advantages = np.empty_like(rewards, dtype=np.float32)
		lastgaelam = 0
		for t in reversed(range(len(rewards))):
			nonterminal = 1 - dones[t]
			delta = rewards[t] + self.gamma * \
				values[t + 1] * nonterminal - values[t]
			lastgaelam = delta + self.gamma * self.lam * nonterminal * lastgaelam
			advantages[t] = lastgaelam

		return advantages

	def save(self, path):
		torch.save(self.policy.state_dict(), path)

	def load(self, path):
		self.policy.load_state_dict(torch.load(path))

	def save_checkpoint(self, path):
		torch.save({
			"model": self.policy.state_dict(),
			"optimizer": self.optimizer.state_dict(),
			"scheduler": self.scheduler.state_dict() if self.anneal_lr else None
		}, path)

	def load_checkpoint(self, path):
		checkpoint = torch.load(path)
		self.policy.load_state_dict(checkpoint["model"])
		self.optimizer.load_state_dict(checkpoint["optimizer"])
		if self.anneal_lr:
			self.scheduler.load_state_dict(checkpoint["scheduler"])

	# make sure there is an extra obs at the end of the trajectory for bootstrapping the TD error
	def train(self, obs, act, old_log_probs, rews, dones, num_updates=10):

		with torch.no_grad():
			# convert to tensors
			obs = torch.from_numpy(obs).float().to(self.device)
			act = torch.from_numpy(act).long().to(self.device)
			old_log_probs = torch.from_numpy(
				old_log_probs).float().to(self.device)

			# bootstrap final reward from trajectory with value estimate, as we don't know what the true final reward would be (since it's the end of the episode)
			values = self.policy.get_value(obs).squeeze(-1)
			rews[-1] += self.gamma * values[-1] * (1 - dones[-1]) # <- note that we dont really need the last 1-done b/c it's recalcualted in the GAE computation but im too lazy to change it

			# advantage computation (I still dont know how GAE works so we pray)
			values = values.cpu().numpy()
			advantages = self.get_advantages(rews, dones, values)
			critic_target = advantages + values[:-1]
			advantages = (advantages - np.mean(advantages)) / \
				(np.std(advantages) + 1e-8)

			# convert to tensors
			advantages = torch.from_numpy(advantages).float().to(self.device)
			critic_target = torch.from_numpy(
				critic_target).float().to(self.device)

			# discard last obs as we don't have a reward for it
			obs = obs[:-1]

		# Update policy whole batch at once, I dont see the benefit in minibatching within a rollout
		for _ in range(num_updates):

			# actor loss
			new_log_probs = self.policy.get_log_probs(obs, act)

			log_ratio = new_log_probs - old_log_probs
			ratio = torch.exp(log_ratio)
			clipped_ratio = torch.clamp(ratio, 1 - self.clip, 1 + self.clip)
			surrogate_loss = - \
				torch.min(ratio * advantages,
						  clipped_ratio * advantages).mean()

			# critic loss
			values = self.policy.get_value(obs).squeeze(-1)
			value_loss = F.mse_loss(values, critic_target)

			# entropy loss (must be under new policy distribution)
			_, _, entropy = self.policy.get_action(obs)
			entropy_loss = -entropy.mean()

			# optimization step
			loss = surrogate_loss + self.value_coeff * \
				value_loss + self.entropy_coeff * entropy_loss

			self.optimizer.zero_grad()
			loss.backward()
			nn.utils.clip_grad_norm_(
				self.policy.parameters(), self.max_grad_norm)
			self.optimizer.step()

			with torch.no_grad():
				# get approximate single-sample KL divergence between old and new policy. Equation taken from http://joschu.net/blog/kl-approx.html
				# kl = (-log_ratio).mean()
				kl = ((ratio - 1) - log_ratio).mean().item() # <- More numerically stable KL computation
				if self.target_kl is not None and kl > self.target_kl:
					break

		return surrogate_loss.item(), value_loss.item(), entropy_loss.item(), kl


def make_env(env_name, render_mode=None, num_envs=4):
	# make atari env
	if render_mode is not None:
		env = gym.make(env_name, render_mode=render_mode)
		return env
	env = make_atari_env(env_name, n_envs=num_envs, seed=0)
	env = VecFrameStack(env, n_stack=4)

	# to make it work with other code
	env.single_observation_space = gym.spaces.Box(
		0, 255, (4, 84, 84), dtype=np.uint8)
	env.single_action_space = env.action_space
	return env


def make_cartpole_env(n_envs):
	env = gym.vector.make("CartPole-v1", num_envs=n_envs)
	return env


def process_obs(obs):
	obs = np.array(obs, dtype=np.float32)
	obs = np.moveaxis(obs, -1, 1)
	obs /= 255.0
	return obs


def main(
		epochs=200,
		rollout_len=512,
		save_every_epochs=10,
		save_path="models/ppo",
		learning_rate=2.5e-4,
		gamma=0.99,
		clip=0.1,
		value_coeff=0.5,
		entropy_coeff=0.01,
		max_grad_norm=0.5,
		gae_lambda=0.95,
		target_kl=None,
		model_params=None,
		logs_path="logs/ppo",
		start_from_epoch=0,
		chkpt_path=None,
		num_envs=4,
		share_extractor=True,
		train_iters=4,
		anneal_lr=True,
):
	# env = make_env("BreakoutNoFrameskip-v4", num_envs=num_envs)
	env = make_cartpole_env(num_envs)
	agent = PPO(
		env.single_observation_space.shape, env.single_action_space.n, lr=learning_rate, gamma=gamma, clip=clip, value_coeff=value_coeff,
		entropy_coeff=entropy_coeff, max_grad_norm=max_grad_norm, target_kl=target_kl, lam=gae_lambda, model_params=model_params,
		share_extractor=share_extractor, anneal_lr=None if not anneal_lr else epochs
	)
	logger = Logger(logs_path)

	if chkpt_path is not None:
		print("Staring from checkpoint: ", chkpt_path)
		print("Starting from epoch: ", start_from_epoch)
		agent.load_checkpoint(chkpt_path)

	logger.log_hps({
		"training_epochs": epochs,
		"rollout_len": rollout_len,
		"learning_rate": learning_rate,
		"gamma": gamma,
		"surrogate_epsillon": clip,
		"value_coeff": value_coeff,
		"entropy_coeff": entropy_coeff,
		"max_gradient": max_grad_norm,
		"lambda": gae_lambda,
		"target_kl": target_kl,
		"num_envs": num_envs,
		"share_extractor": share_extractor,
		"gradient_updates_per_rollout": train_iters,
		"anneal_lr": anneal_lr,
		"start_from_epoch": start_from_epoch,
		"chkpt_path": chkpt_path,
		"save_every_epochs": save_every_epochs,
		"save_path": save_path,
	})

	obs, _ = env.reset()
	done = False
	# obs = process_obs(obs)
	obs = np.array(obs)

	for epoch in range(start_from_epoch, epochs):
		# init buffers
		obs_buf = np.ndarray((rollout_len + 1, num_envs, *
							  env.single_observation_space.shape), dtype=np.float32)
		act_buf = np.ndarray(
			(rollout_len, num_envs, *env.single_action_space.shape), dtype=np.int32)
		old_log_probs_buf = np.ndarray(
			(rollout_len, num_envs, *env.single_action_space.shape), dtype=np.float32)
		rews_buf = np.ndarray((rollout_len, num_envs), dtype=np.float32)
		dones_buf = np.ndarray((rollout_len, num_envs), dtype=np.float32)

		# metrics
		ep_rews = np.zeros(num_envs)
		ep_len = np.zeros(num_envs)

		avg_ep_len = []
		avg_ep_rew = []

		for t in range(rollout_len):
			with torch.no_grad():
				# step
				act, log_prob, _ = agent.policy.get_action(
					torch.from_numpy(obs).float().to(agent.device)
				)
				act = act.cpu().numpy()
				next_obs, rew, done, _, _ = env.step(act)

				# add to buffer
				obs_buf[t] = obs
				act_buf[t] = act
				old_log_probs_buf[t] = log_prob.cpu().numpy()
				rews_buf[t] = rew
				dones_buf[t] = done

				# update obs
				# obs = process_obs(next_obs)
				obs = np.array(next_obs)

				# Metrics
				ep_len += 1
				ep_rews += rew

				for i, d in enumerate(done):
					if d:
						avg_ep_len.append(ep_len[i])
						ep_len[i] = 0
						avg_ep_rew.append(ep_rews[i])
						ep_rews[i] = 0

		obs_buf[-1] = obs

		avg_ep_len.extend(ep_len)
		avg_ep_rew.extend(ep_rews)

		# transpose buffers to be of shape (num_envs, ep_len, ...)
		obs_buf = obs_buf.swapaxes(0, 1)
		act_buf = act_buf.swapaxes(0, 1)
		old_log_probs_buf = old_log_probs_buf.swapaxes(0, 1)
		rews_buf = rews_buf.swapaxes(0, 1)
		dones_buf = dones_buf.swapaxes(0, 1)

		# train
		avg_surrogate_loss = 0
		avg_value_loss = 0
		avg_entropy_loss = 0
		avg_kl = 0

		# Minibatching aross envs
		for i in range(num_envs):
			surrogate_loss, value_loss, entropy_loss, kl = agent.train(
				obs_buf[i], act_buf[i], old_log_probs_buf[i], rews_buf[i], dones_buf[i], num_updates=train_iters)

			avg_surrogate_loss += surrogate_loss
			avg_value_loss += value_loss
			avg_entropy_loss += entropy_loss
			avg_kl += kl
		
		# anneal learning rate after each epoch
		if anneal_lr:
			agent.scheduler.step()

		avg_surrogate_loss /= num_envs
		avg_value_loss /= num_envs
		avg_entropy_loss /= num_envs
		avg_kl /= num_envs

		# log
		logger.log_scalar("loss/surrogate_loss", avg_surrogate_loss, epoch)
		logger.log_scalar("loss/value_loss", avg_value_loss, epoch)
		logger.log_scalar("loss/entropy_loss", avg_entropy_loss, epoch)
		logger.log_scalar("loss/kl", avg_kl, epoch)
		logger.log_scalar("rollout/avg_ep_len", np.mean(avg_ep_len), epoch)
		logger.log_scalar("rollout/avg_ep_rew", np.mean(avg_ep_rew), epoch)
		logger.log_scalar("lr", agent.optimizer.param_groups[0]["lr"], epoch)

		print("epoch: ", epoch, "avg_ep_len: ", np.mean(avg_ep_len), "avg_ep_rew: ", np.mean(avg_ep_rew), "avg_surrogate_loss: ",
			  avg_surrogate_loss, "avg_value_loss: ", avg_value_loss, "avg_entropy_loss: ", avg_entropy_loss, "avg_kl: ", avg_kl)

		# save
		if epoch % save_every_epochs == 0:
			agent.save_checkpoint(f"{save_path}_{epoch}.pth")

	agent.save(f"{save_path}.pth")
	env.close()


def eval():
	# env = make_env("BreakoutNoFrameskip-v4", render_mode="human")
	env = gym.make("CartPole-v1", render_mode="human")
	agent = PPO(env.observation_space.shape,
				env.action_space.n, share_extractor=True)
	agent.load("models/ppo.pth")

	obs, _ = env.reset()
	obs = np.array(obs)
	obs = np.expand_dims(obs, axis=0)
	done = False
	while not done:
		act, _, _ = agent.policy.get_action(
			torch.from_numpy(obs).float().to(agent.device))
		next_obs, rew, done, _, _ = env.step(act.detach().item())
		obs = np.array(next_obs)
		obs = np.expand_dims(obs, axis=0)
		env.render()

if __name__ == "__main__":
	# main(logs_path="logs/ppo_cartpole_3")
	eval()
	# test()
