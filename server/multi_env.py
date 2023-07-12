import multi_instance_manager as mim
import websockets
import asyncio
import json
import collections


class MultiEnv:
    def __init__(self, n_env=4, render_colored=False, observation_size=(100, 57), action_size=81, frames_per_wait=5, server_ip="localhost", server_port='8080', time_scale=1, level="GG_Hornet_1", hk_data_dir="Hollow Knight_Data", hk_path="F:\GOG Games\Hollow Knight", hk_save_dir="C:\\Users\\adity\\AppData\\LocalLow\\Team Cherry\\Hollow Knight"):
        self.n_env = n_env
        self.server_ip = server_ip
        self.server_port = server_port
        self.obs_size = observation_size
        self.action_size = action_size
        self.frame_delay = frames_per_wait
        self.time_scale = time_scale
        self.level = level

        self.render_mode = 'human'
        self.messageQueue = [collections.deque() for _ in range(self.n_env)]
        self.websockets = [None for _ in range(self.n_env)]
        self.connected = [asyncio.Event() for _ in range(self.n_env)]
        self.augment_images = render_colored

        # self.ws_task = asyncio.create_task(self._start_server())

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
            
        self.instance_manager.spawn_n(self.n_env)
        self.instance_manager.start_all()

    async def _start_server(self):
        await websockets.serve(self._on_connect, self.server_ip, self.server_port)

    def close(self):
        self.instance_manager.kill_all()
        self.instance_manager.destroy_all()
        # self.instance_manager.restore_old_saves("recent")

    # async def _on_connect(self, websocket, path):


env = MultiEnv()
wait = input("Press enter to continue")
env.close()
