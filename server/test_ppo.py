from ppo import PPO
from multi_env import MultiEnv
import torch
import asyncio

import numpy as np

async def make_env(timescale, frames_per_wait, n_envs=4):
	# env = gym.make('Pong-ramNoFrameskip-v4')

	# env = atari_preprocessing.AtariPreprocessing(env)
	# env = gym.wrappers.FrameStack(env, 4)
	# return env
	env = MultiEnv(n_env=n_envs, render_colored=False, time_scale=timescale,
				   frames_per_wait=frames_per_wait, level="GG_Mega_Moss_Charger", pause_after_step=False)
	await env.loadAll()
	return env


async def test_all_saves():
	# Create a new environment
	env = await make_env(timescale=1, frames_per_wait=1, n_envs=1)
	# Create a new PPO agent
	obs_shape = (1, env.obs_size[0], env.obs_size[1])
	agent = PPO(
		obs_size=obs_shape, act_size=env.action_size, lr=2.5e-4, gamma=0.99, clip=0.1, value_coeff=0.5,
		entropy_coeff=0.01, max_grad_norm=0.5, target_kl=0.1, lam=0.95, model_params=None,
		share_extractor=True, anneal_lr=None if not True else 400
	)
	for i in range(0, 400, 50):
		# Load the checkpoint
		agent.load_checkpoint("models/hollow_knight/ppo_mosscharger/_" + str(i) + ".pth")
		# Run the environment
		await env.resume(0)
		obs = await env.reset(0)
		obs = np.expand_dims(obs, axis=0)
		done = False
		total_reward = 0.0
		while not done:
			# Get the action from the agent
			act, _, _ = agent.policy.get_action(torch.from_numpy(obs).float().to(agent.device))
			# Run the environment
			obs, reward, done, info = await env.step(act.item(), 0)
			obs = np.expand_dims(obs, axis=0)
			if reward > 0.0:
				print("Reward: ", reward)
			total_reward += reward
		await env.pause(0)
		
		print("Save_from_epoch: ", i, "Reward: ", total_reward)
		
	await env.close()

asyncio.run(test_all_saves())