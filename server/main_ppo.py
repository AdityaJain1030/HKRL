from ppo import PPO
from multi_env import MultiEnv
from logger import Logger

import numpy as np
import torch
import asyncio


async def make_env(timescale, frames_per_wait, n_envs=4):
	# env = gym.make('Pong-ramNoFrameskip-v4')

	# env = atari_preprocessing.AtariPreprocessing(env)
	# env = gym.wrappers.FrameStack(env, 4)
	# return env
	env = MultiEnv(n_env=n_envs, render_colored=False, time_scale=timescale,
				   frames_per_wait=frames_per_wait, level="GG_Mega_Moss_Charger", pause_after_step=False)
	await env.loadAll()
	return env


async def main(
	epochs=400,
	rollout_len=4096,
	save_every_epochs=50,
	save_path="models/ppo_mosscharger/",
	learning_rate=2.5e-4,
	gamma=0.99,
	clip=0.1,
	value_coeff=0.5,
	entropy_coeff=0.01,
	max_grad_norm=0.5,
	gae_lambda=0.95,
	target_kl=0.1,
	model_params=None,
	logs_path="logs/ppo_mosscharger_1",
	start_from_epoch=0,
	chkpt_path=None,
	num_envs=6,
	share_extractor=True,
	train_iters=10,
	anneal_lr=True,
	log_steps=False,
	frames_per_wait=1,
	timescale=1,
):
	env = await make_env(timescale=timescale, frames_per_wait=frames_per_wait, n_envs=num_envs)
	obs_shape = (1, env.obs_size[0], env.obs_size[1])
	agent = PPO(
		obs_size=obs_shape, act_size=env.action_size, lr=learning_rate, gamma=gamma, clip=clip, value_coeff=value_coeff,
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
	},
	[
		"loss/surrogate_loss",
		"loss/value_loss",
		"loss/entropy_loss",
		"rollout/avg_ep_rew",
		"metrics/lr",
		"metrics/clip_frac",
		"metrics/explained_variance",
		"metrics/kl"
	])

	for epoch in range(start_from_epoch, epochs):
		obs = await env.resetall()

		# init buffers
		# TODO: Make this more memory efficient (initialize minibatch instead of whole batch, allows for greater rollout and maybe greater performance?)
		obs_buf = np.ndarray((rollout_len + 1, num_envs, *
							  obs_shape), dtype=np.float32)
		act_buf = np.ndarray(
			(rollout_len, num_envs), dtype=np.int32)
		old_log_probs_buf = np.ndarray(
			(rollout_len, num_envs), dtype=np.float32)
		rews_buf = np.ndarray((rollout_len, num_envs), dtype=np.float32)
		dones_buf = np.ndarray((rollout_len, num_envs), dtype=np.float32)

		# metrics
		ep_rews = np.zeros(num_envs)
		avg_ep_rew = []

		for t in range(rollout_len):
			with torch.no_grad():
				#step
				act, log_prob, _ = agent.policy.get_action(
					torch.from_numpy(obs).float().to(agent.device)
				)
				act = act.cpu().numpy()
				next_obs, rew, _, _ = await env.stepall(act.tolist())

				# add to buffer
				obs_buf[t] = obs
				act_buf[t] = act
				old_log_probs_buf[t] = log_prob.cpu().numpy()
				rews_buf[t] = rew
				dones_buf[t] = [False for _ in range(num_envs)] # Environment has no good done signal

				obs = next_obs

				# update metrics
				ep_rews += rew

		# for bootsrapping		
		obs_buf[-1] = obs

		# This is a bit jank, as during the end of training where the AI only dies once or twice,
		# the avg ep length and reward will look low, as one death will 
		# half the average, but the AI is still doing well. 
		# To my knowledge, this shouldnt affect training, but it's something to keep in mind
		# On the reward graph this has the most effect during the end of training,
		# so crank up the smoothing to see the trend for last few epochs
		avg_ep_rew = np.mean(ep_rews)

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
		clip_frac = 0

		# Minibatching aross envs is easier than across rollouts, and almost as good
		await env.pauseall() # <- environment will keep running, pausing to lower resource usage

		for i in range(num_envs):
			surrogate_loss, value_loss, entropy_loss, kl, clip_frac, ev = agent.train(
			obs_buf[i], act_buf[i], old_log_probs_buf[i], rews_buf[i], dones_buf[i], num_updates=train_iters)

			avg_surrogate_loss += surrogate_loss
			avg_value_loss += value_loss
			avg_entropy_loss += entropy_loss
			avg_kl += kl
			clip_frac += clip_frac

		# anneal learning rate after each epoch
		if anneal_lr:
			agent.scheduler.step()
		
		# log
		avg_surrogate_loss /= num_envs
		avg_value_loss /= num_envs
		avg_entropy_loss /= num_envs
		avg_kl /= num_envs
		clip_frac /= num_envs

		if log_steps:
			t = epoch * num_envs * rollout_len
		else:
			t = epoch

		logger.log_scalar("loss/surrogate_loss", avg_surrogate_loss, t) # <- Graph should be decreasing
		logger.log_scalar("loss/value_loss", avg_value_loss, t) # <- Graph should be approaching 0 from positive
		logger.log_scalar("loss/entropy_loss", avg_entropy_loss, t) # <- Graph should be approaching to 0 from negative

		# theres a few problems with how I implemented these metrics so if the agent looks varied at the end of training, this is probably why (see above)
		logger.log_scalar("rollout/avg_ep_rew", np.mean(avg_ep_rew), t) # <- Graph should be increasing, main metric

		logger.log_scalar("metrics/lr", agent.optimizer.param_groups[0]["lr"], t)
		logger.log_scalar("metrics/clip_frac", clip_frac, t)
		logger.log_scalar("metrics/explained_variance", ev, t) # <- Should be approaching 1
		logger.log_scalar("metrics/kl", avg_kl, t) # <- Should stay under 0.1, hopefully under 0.05

		logger.log_scalar("other/num_grad_updates", epoch * num_envs * train_iters, t)
		logger.log_scalar("other/num_timesteps", epoch * num_envs * rollout_len, t)

		print("epoch: ", epoch, "avg_ep_rew: ", np.mean(avg_ep_rew), "avg_surrogate_loss: ",
			  avg_surrogate_loss, "avg_value_loss: ", avg_value_loss, "avg_entropy_loss: ", avg_entropy_loss, "avg_kl: ", avg_kl)

		# save
		if epoch % save_every_epochs == 0:
			agent.save_checkpoint(f"{save_path}_{epoch}.pth")

		await env.resumeall()


asyncio.run(main())