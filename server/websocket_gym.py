import gymnasium as gym
from gymnasium import spaces

import numpy as np
import websockets
import asyncio
import json
import collections

# logger = logging.getLogger('websockets')
# logger.setLevel(logging.DEBUG)
# logger.addHandler(logging.StreamHandler())

class WebsocketEnv:
	metadata = {'render.modes': ['human']}

	def __init__(self, render_colored=False, observation_size=(100, 57), action_size = 81, frames_per_wait = 5, server_ip='localhost', server_port='8080', time_scale=1, level = "GG_Hornet_1"):
		self.server_ip = server_ip
		self.server_port = server_port
		self.obs_size = observation_size
		self.action_size = action_size
		self.frame_delay = frames_per_wait
		self.time_scale = time_scale
		self.level = level

		self.render_mode = 'human'
		self.messageQueue = collections.deque()
		self.websocket = None
		self.connected = asyncio.Event()
		self.augment_images = render_colored

		self.ws_task = asyncio.create_task(self._start_server())

	async def _start_server(self):
		await websockets.serve(self._on_connect, self.server_ip, self.server_port)

	async def _on_connect(self, websocket, path):
		print("Connected to client")
		self.websocket = websocket if self.websocket is None else self.websocket
		self.connected.set()
		await asyncio.Future()

	async def _wait_for_connection(self):
		await self.connected.wait()

	def cvtclr(self, num):
		if num == 0:
			return (0, 0, 0)
		elif num == 1:
			return (255, 0, 0)
		elif num == 2:
			return (0, 255, 0)
		elif num == 3:
			return (0, 0, 255)
		elif num == 4:
			return (255, 255, 255)

	async def reset(self):
		await self._wait_for_connection()
		message = await self.sendMessage('init',  {
			'state_size': self.obs_size,
			'action_size': self.action_size - 1
,
			'level': self.level,
			'frames_per_wait': self.frame_delay,
			'time_scale': self.time_scale
			
		})

		message = json.loads(message)
		# obs = np.asarray(message['data']['state'], dtype=np.uint8)
		if self.augment_images:
			obs = np.asarray([self.cvtclr(i) for i in message['data']['state']], dtype=np.uint8)
			obs = obs.reshape((self.obs_size[0], self.obs_size[1], 3))
		else:
			obs = np.asarray(message['data']['state'], dtype=np.float16)
			obs = obs.reshape(self.obs_size[0], self.obs_size[1])
			obs /= 4

		obs = np.rot90(obs)

		return obs
		# print(json.loads(message))

	async def sendMessage(self, type, data):
		message = json.dumps({'type': type, 'data': data, 'sender': 'server'})
		await self.websocket.send(message)
		message = await self.websocket.recv()
		return message
				
	async def step(self, action):
		await self._wait_for_connection()
		message = await self.sendMessage('action', {'action': action})
		message = json.loads(message)

		if self.augment_images:
			obs = np.asarray([self.cvtclr(i) for i in message['data']['state']], dtype=np.uint8)
			obs = obs.reshape((self.obs_size[0], self.obs_size[1], 3))
		else:
			obs = np.asarray(message['data']['state'], dtype=np.float16)
			obs = obs.reshape(self.obs_size[0], self.obs_size[1])
			obs /= 4

		obs = np.rot90(obs)

		return obs, message['data']['reward'], message['data']['done'], message['data']['info']

	def close(self):
		self.websocket.close()
		self.ws_task.cancel()
		self.connected.clear()
		print("Websocket closed")
	
	async def pause(self):
		await self._wait_for_connection()
		await self.sendMessage('pause', {})
		return
	
	async def resume(self):
		await self._wait_for_connection()
		await self.sendMessage('resume', {})
		return