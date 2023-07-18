import cv2
import numpy as np
import matplotlib.pyplot as plt

from multi_env import MultiEnv
import torch
import random
import asyncio
import datetime

import gymnasium as gym
from gymnasium.wrappers import atari_preprocessing, frame_stack
from CnnDQN import CnnDQN
from LinearDQN import LinearDQN
from replaybuffer import ReplayBuffer

from logger import Logger


def preprocess(obs):
	# obs = np.asarray(obs)
	return np.expand_dims(obs, 0)
	# return np.expand_dims(obs, 0)


async def make_env(timescale, frames_per_wait):
	# env = gym.make('Pong-ramNoFrameskip-v4')

	# env = atari_preprocessing.AtariPreprocessing(env)
	# env = gym.wrappers.FrameStack(env, 4)
	# return env
	env = MultiEnv(n_env=1, render_colored=False, time_scale=timescale,
				   frames_per_wait=frames_per_wait, level="GG_Mega_Moss_Charger")
	await env.load(0)
	return env


async def main(
	init_eplison=0.95,
	final_eplison=0.05,
	e_greedy_steps=10000,
	lr=0.0001,
	gamma=0.99,
	batch_size=32,
	buffer_size=50000,
	update_target_every=16,
	learning_starts=10000,
	soft_update_every=2,
	train_every=4,
	train_timesteps=1000000,
	save_every=10000,
	max_timesteps=2000,
	save_path='./models/pong',
	log_path='./logs/pong',
	time_scale=1,
	frames_per_wait=1,
	load_model = None

):
	# env = gym.vector.SyncVectorEnv([make_env for _ in range(8)])
	env = await make_env(timescale=time_scale, frames_per_wait=frames_per_wait)
	# print("action space", env.action_space)

	# agent_nopool = CnnDQN(env.observation_space.shape, (env.action_space.n,), lr=lr, gamma=gamma)
	# agent_pool = CnnDQN(env.observation_space.shape, env.action_space.n, lr=lr, gamma=gamma, use_pooling=True)
	agent_linear = CnnDQN(
		(1, env.obs_size[0], env.obs_size[1]), (env.action_size,), lr=lr, gamma=gamma)
	buffer = ReplayBuffer(buffer_size)
	logger = Logger(log_path)
	
	if load_model is not None:
		print("loading model: " + load_model)
		agent_linear.load(load_model)

	# action_list = ["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"]

	eplison = init_eplison
	obs = await env.reset(0)
	obs = preprocess(obs)
	episode = 0

	# metrics
	ep_len = 0
	ep_reward = []
	ep_loss = []
	ep_actions = [0 for _ in range(env.action_size)]
	ep_sample_age = []

	for t in range(train_timesteps):

		# training loop
		if t % update_target_every == 0:
			agent_linear.synchronize_target()
			# agent_pool.synchronize_target()

		# if t % soft_update_every == 0:
		# 	agent_nopool.synchronize_target(polyak=)
			# agent_pool.soft_update()

		action = agent_linear.get_actions(obs, eplison)
		# action = agent_pool.get_action(obs, eplison)

		next_obs, reward, done, _ = await env.step(action[0], 0)
		next_obs = preprocess(next_obs)
		buffer.push(obs, action[0], reward, next_obs, done)
		obs = next_obs

		ep_len += 1
		ep_reward.append(reward)
		ep_actions[action[0]] += 1

		if t > learning_starts:
			if t % train_every == 0:
				await env.pause(0)
				s, a, r, s_n, d, i = buffer.sample(batch_size)
				if t > buffer_size:
					i = [age + t for age in i]

				ep_sample_age.extend(i)

				loss = agent_linear.train(s, a, r, s_n, d)
				ep_loss.append(loss)
				await env.resume(0)

			eplison = max(final_eplison, init_eplison - (init_eplison -
														 final_eplison) * (t - learning_starts) / e_greedy_steps)

			if t % save_every == 0:
				await env.pause(0)
				agent_linear.save(save_path + "_cnn" + str(t) + ".pt")
				await env.resume(0)
				# agent_pool.save(save_path + "_pool")

		if ep_len > max_timesteps:
			obs = await env.reset(0)
			obs = preprocess(obs)
			print("episode: ", episode, "ep_len: ", ep_len, "ep_reward: ", sum(ep_reward), "avg_loss: ", np.mean(ep_loss), "eplison: ", eplison)
			logger.log_scalar("ep_len", ep_len, episode)
			logger.log_scalar("total_rew", np.sum(ep_reward), episode)
			logger.log_scalar("avg_loss", np.mean(ep_loss), episode)
			logger.log_scalar("eplison", eplison, episode)
			# logger.log_barplot("ep_actions", action_list, ep_actions, episode)
			logger.log_scalar("avg_sample_age", np.mean(ep_sample_age), episode)

			ep_len = 0
			ep_reward = []
			ep_loss = []
			ep_actions = [0 for _ in range(env.action_size)]
			ep_sample_age = []

			episode += 1
	env.close()


# Adapted atari hyperparams 
asyncio.run(main(
	log_path="./logs/hollow_knight/mossbag" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
	train_every=4,
	train_timesteps=1000000,
	update_target_every=1000,
	learning_starts=100000,
	e_greedy_steps=100000,
	save_every=50000,
	max_timesteps=5000, #half of atari b/c enemies seem to disappear after a while
	buffer_size=500000,
	gamma=0.995,
	time_scale=5,
	frames_per_wait=1,
	batch_size=32,
	lr=0.0001,
	# load_model="./models/hollow_knight/mossbag_cnn60000.pt",
	save_path="./models/hollow_knight/mossbag2",
	# init_eplison=0.65, #start from pretrained model
	))
