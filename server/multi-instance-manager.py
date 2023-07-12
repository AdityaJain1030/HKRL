import os, fnmatch, shutil, subprocess

# dataDir = "Hollow Knight_Data"
# steamAppId = "steam_appid.txt"

# dir = "F:\GOG Games\Hollow Knight"

class MultiInstanceManager:
	def __init__(self, hollow_knight_dir, data_dir):
		self.root = hollow_knight_dir
		self.data_dir = data_dir
		self.steam_app_id = os.path.join(self.root, "steam_appid.txt")

		self.exe = None
		for _, _, files in os.walk(self.root):
			for name in files:
				if fnmatch.fnmatch(name, "hollow knight.*"):
					self.exe = os.path.join(self.root, name)
					break
		if self.exe is None:
			print("Could not find HK exe, nothing will work")
			return

		self.instances = []
		self._processes = []

	
	def check_for_exe(self):
		if self.exe is None:
			return False
		return True
	
	def _get_instance_exe_name(self, instance_name):
		if not self.check_for_exe():
			return None
		return self.exe.replace("hollow knight", instance_name)

	def _get_instance_data_name(self, instance_name):
		if not self.check_for_exe():
			return None
		return os.path.join(self.root, instance_name + "_Data")
	
	def _instance_exists(self, name):
		if not self.check_for_exe():
			return False
		return os.path.exists(self._get_instance_data_name(name)) or os.path.exists(self._get_instance_exe_name(name))
	
	def create_instance(self, name):		
		if not self.check_for_exe():
			return False
		
		if self._instance_exists(name):
			return False
		
		try:
			subprocess.check_call('mklink /J "%s" "%s"' % (self._get_instance_data_name(name), os.path.join(self.root, self.data_dir)), shell=True)
			f = open(self.steam_app_id, "w")
			f.write("367520")
			f.close()

			shutil.copyfile(self.exe, self._get_instance_exe_name(name))
			self.instances.append(name)
			return True
		except:
			return False
		
	def delete_instance(self, name):
		if not self.check_for_exe():
			return False
		
		if not self._instance_exists(name):
			return False
		
		try:
			os.remove(self._get_instance_exe_name(name))
			os.remove(self._get_instance_data_name(name))
			self.instances.remove(name)
			return True
		except:
			return False
	
	def delete_all_instances(self):
		if not self.check_for_exe():
			return False
		
		for instance in self.instances:
			self.delete_instance(instance)
		return True
	
	def start_instance(self, name):
		if not self.check_for_exe():
			return False
		
		if not self._instance_exists(name):
			return False
		
		try:
			self._processes.append((name, subprocess.Popen(self._get_instance_exe_name(name))))
			return True
		except:
			return False
		
	def end_instance(self, name):
		if not self.check_for_exe():
			return False
		
		if not self._instance_exists(name):
			return False
		
		try:
			for process in self._processes:
				if process[0] == name:
					process[1].terminate()
					self._processes.remove(process)
					process[1].wait()
					return True
			return False
		except:
			return False
	

		

# manager = MultiInstanceManager(dir, dataDir)
# manager.create_instance("test")
# manager.start_instance("test")
# wait = input("Press enter to continue")
# manager.end_instance("test")
# manager.delete_instance("test")