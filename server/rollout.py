import asyncio
import websocket_gym
import dqn
import random
import cv2
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import spaces
import torch

# import seaborn as sns; sns.set()


async def main():
	env = websocket_gym.WebsocketEnv(
		level="GG_Hornet_1", time_scale=3, frames_per_wait=2)
	obs = await env.reset()
	rew_tot = 0
	i = 0
	now = datetime.now()
	# await asyncio.sleep(1)
	while True:
		i += 1
	# print(obs[:,:,0])
		# print(i)
		if i % 100 == 0:
			print(f"{i} steps in: {(datetime.now() - now).total_seconds()} seconds")
			now = datetime.now()
		# 	await env.pause()
		# 	await asyncio.sleep(5)
		# 	await env.resume()
		obs, rew, done, _ = await env.step(random.randint(0, 80))
		rew_tot += rew
		imS = cv2.resize(obs, (960, 540))
		cv2.imshow('image', imS)
		cv2.waitKey(1)
		if done:
			print("Done, total reward:"	, rew_tot)
			break


def train_DQN(initial_eplison=1,
					final_eplison=0.05,
					e_greedy_frames = 1000000,
					use_pooling=False,
					lr=0.0001,
					gamma=0.99,
					learning_starts_at=50000,
					training_timesteps=5000000,
					minibatch_size=32,
					update_every=4,
					hardsync_target_every=10000,
					softsync_target_every=None,
					polyak_ratio=1,
					gradient_updates_per_update=1,
					level = "GG_Hornet_1",
					time_scale = 3,
					frames_per_wait = 2
					):
	env = gym.make('BreakoutNoFrameskip-v4')
	env = gym.wrappers.AtariPreprocessing(env)

	agent = dqn.DQN((1, env.observation_space.shape[0], env.observation_space.shape[1]), env.action_space.n, learning_rate=lr, gamma=gamma, use_pooling=use_pooling)
	
	returns = []
	episode_rewards = 0
	
	obs = env.reset()
	obs = np.expand_dims(obs[0], axis=0)
	eplison = initial_eplison

	for timestep in range(training_timesteps):
		if timestep % hardsync_target_every == 0:
			agent.synchronize_target()

		if timestep % 5000 == 0:
			print(f"timestep: {timestep}, eplison: {eplison}, avg_return: {np.mean(returns)}")

		if softsync_target_every is not None and timestep % softsync_target_every == 0:
			agent.soft_synchronize_target(polyak_ratio)

		if random.random() > eplison and timestep > learning_starts_at:
			action = agent.get_action(obs)
		else:
			action = env.action_space.sample()

		if timestep > learning_starts_at:
			eplison -= (initial_eplison - final_eplison) / e_greedy_frames
			eplison = max(eplison, final_eplison)
		
		next_obs, reward, done, _, _ = env.step(action)
		next_obs = np.expand_dims(next_obs, axis=0)

		agent.save_experience(obs, action, reward, next_obs, done)


		next_obs = obs

		episode_rewards += reward

		if timestep > learning_starts_at and timestep % update_every == 0:
			for _ in range(gradient_updates_per_update):
				s, a, r, d, s_n = agent.sample_experience(minibatch_size)
				loss = agent.update(s, a, r, d, s_n)

			if timestep % 10000 == 0:
				print(f"timestep: {timestep}, eplison: {eplison}, avg_return: {np.mean(returns)}, loss: {loss.item()}")


		if done:
			returns.append(episode_rewards)
			episode_rewards = 0
			obs = env.reset()
			obs = np.expand_dims(obs[0], axis=0)

	torch.save(agent.state_dict(), "model.pt")
	plt.title("Returns")
	plt.plot(returns)
	plt.show()

	terminated = False
	env = gym.make('BreakoutNoFrameskip-v4', render_mode="human")
	env = gym.wrappers.AtariPreprocessing(env)
	obs = env.reset()
	while not terminated:
		obs = np.expand_dims(obs[0], axis=0)

		act = agent.get_action(obs)
		next_obs, reward, done, _, _ = env.step(act)

		terminated = done
		# env.re


			# print(f"timestep: {timestep}")
	

	
train_DQN()

# asyncio.run(main())
