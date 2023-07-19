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



async def pp(e):
	return *[i for i in range(e)],

async def run_while():
	data = await asyncio.gather(pp(2), pp(2), pp(2))

	print(*list(zip(*data)))


asyncio.run(run_while())
# async def fill_buffer(env, )