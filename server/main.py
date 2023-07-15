import cv2
import numpy as np
import matplotlib.pyplot as plt

from multi_env import MultiEnv
import torch
import random
import asyncio

envs = 8

async def main():
	# print([i for i in range(1)])
	env = MultiEnv(n_env=envs, render_colored=True, time_scale=5, frames_per_wait=1)
	things = await env.loadAll()
	print(things)
	# wait = input("Press enter to continue")

	obs = await asyncio.gather(*[env.reset(i) for i in range(envs)])
	a = np.concatenate(obs, axis=0)
	for i in range(1000):
		obs, reward, done, info = list(map(list, zip(*await asyncio.gather(*[env.step(random.randint(0, 80), i) for i in range(envs)]))))
		a = np.concatenate(obs, axis=0)
		a = cv2.resize(a, (a.shape[1] * 2, a.shape[0] * 2))
		cv2.imshow('image', a)
		cv2.waitKey(1)

	await env.close()

# async def rollout_task(env, id, ):


asyncio.run(main())
