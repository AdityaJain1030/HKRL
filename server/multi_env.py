import multi_instance_manager as mim
import websockets
import asyncio
import json
import collections
import numpy as np
import cv2

class MultiEnv:
	def __init__(self, n_env=4, render_colored=False, observation_size=(100, 57), action_size=81, frames_per_wait=5, server_ip="localhost", server_port='8080', time_scale=1, level="GG_Hornet_1", hk_data_dir="Hollow Knight_Data", hk_path="F:\GOG Games\Hollow Knight", pause_after_step=False):
		self.n_env = n_env
		self.server_ip = server_ip
		self.server_port = server_port
		self.obs_size = observation_size
		self.action_size = action_size
		self.frame_delay = frames_per_wait
		self.time_scale = time_scale
		self.level = level
		self.pause_after_step = pause_after_step

		self.render_mode = 'human'
		self.websockets = [None for _ in range(self.n_env)]
		self.connected = [asyncio.Event() for _ in range(self.n_env)]
		self.augment_images = render_colored

		# Hollow Knight specific
		self.data_dir = hk_data_dir
		self.hk_path = hk_path
		# self.save_dir = hk_save_dir

		self.instance_manager = mim.MultiInstanceManager(
			self.hk_path, self.data_dir)

		# self.instance_manager.hide_old_saves()
		# for i in range(self.n_env):
		#     self.instance_manager.copy_save(
		#         'F:\\Downloads\\Hollow Knight Mods\\HKRL\\SaveFiles\\user4.dat', i + 1)

		self.ws_task = asyncio.create_task(self._start_server())

		self.instance_manager.spawn_n(self.n_env)
		self.instance_manager.start_all()

	async def _start_server(self):
		await websockets.serve(self._on_connect, self.server_ip, self.server_port)

	async def _on_connect(self, websocket, path):
		print("Connected to client")
		self.websockets[self.websockets.index(None)] = websocket

		self.connected[self.websockets.index(websocket)].set()
		await asyncio.Future()

	def _cleanup_connection(self, env_id):
		self.websockets[env_id]
		self.connected[env_id].clear()

	async def close(self):
		for i in self.websockets:
			if i is not None:
				await i.close()
		self.ws_task.cancel()
		self.connected.clear()
		self.instance_manager.kill_all()
		# self.instance_manager.destroy_all()

	async def sendMessage(self, type, data, env_id):
		message = json.dumps({'type': type, 'data': data, 'sender': 'server'})
		try:
			await self.websockets[env_id].send(message)
			message = await self.websockets[env_id].recv()
			return message
		except websockets.exceptions.ConnectionClosed:
			print("Connection closed")
			self._cleanup_connection(env_id)
			return None

	async def reset(self, env_id):
		await self.connected[env_id].wait()
		message = message = await self.sendMessage('reset',  {
			'state_size': self.obs_size,
			'action_size': self.action_size - 1,
			'level': self.level,
			'frames_per_wait': self.frame_delay,
			'time_scale': self.time_scale
		}, env_id)
		if message is None:
			return None
		message = json.loads(message)
		
		if self.augment_images:
			obs = np.asarray([self.cvtclr(i) for i in message['data']['state']], dtype=np.uint8)
			obs = obs.reshape((self.obs_size[0], self.obs_size[1], 3))
		else:
			obs = np.asarray(message['data']['state'], dtype=np.float16)
			obs = obs.reshape(self.obs_size[0], self.obs_size[1])
			obs /= 4
			obs *= 255

		obs = np.rot90(obs)
		obs = np.expand_dims(obs, axis=0)


		return obs
	
	async def step(self, action, env_id):
		await self.connected[env_id].wait()

		if self.pause_after_step:
			await self.resume(env_id)

		message = await self.sendMessage('action', {'action': action}, env_id)
		
		if self.pause_after_step:
			await self.pause(env_id)

		if message is None:
			return None, None, None, None
		
		message = json.loads(message)
		if self.augment_images:
			obs = np.asarray([self.cvtclr(i) for i in message['data']['state']], dtype=np.uint8)
			obs = obs.reshape((self.obs_size[0], self.obs_size[1], 3))
		else:
			obs = np.asarray(message['data']['state'], dtype=np.float16)
			obs = obs.reshape(self.obs_size[0], self.obs_size[1])
			obs /= 4
			obs *= 255

		obs = np.rot90(obs)
		obs = np.expand_dims(obs, axis=0)

		return obs, message['data']['reward'], message['data']['done'], message['data']['info']
	
	async def stepall(self, actions):
		data = await asyncio.gather(*[self.step(actions[i], i) for i in range(self.n_env)])
		return (*map(np.array, zip(*data)),)
	
	async def resetall(self):
		data = await asyncio.gather(*[self.reset(i) for i in range(self.n_env)])
		return np.array(data)

	async def pause(self, env_id):
		await self.connected[env_id].wait()
		message = await self.sendMessage('pause', {}, env_id)
		if message is None:
			return None
		return True
	
	async def pauseall(self):
		await asyncio.gather(*[self.pause(i) for i in range(self.n_env)])
		return True
	
	async def resumeall(self):
		await asyncio.gather(*[self.resume(i) for i in range(self.n_env)])
		return True
	
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
	
	async def resume(self, env_id):
		await self.connected[env_id].wait()
		message = await self.sendMessage('resume', {}, env_id)
		if message is None:
			return None
		return True

	async def loadAll(self):
		# loaded = []
		# for i in range(self.n_env):
		# 	loaded.append(await self.load(i))
		loaded = await asyncio.gather(*[self.load(i) for i in range(self.n_env)])
		return loaded
	
	async def load(self, env_id):
		await self.connected[env_id].wait()
		message = await self.sendMessage('init', {}, env_id)
		if message is None:
			return None
		return True
