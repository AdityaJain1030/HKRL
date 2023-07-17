import os
import fnmatch
import shutil
import subprocess
import psutil

# Inspiration from 
class MultiInstanceManager:
	def __init__(self, hollow_knight_dir, data_dir):
		self.root = hollow_knight_dir
		self.data_dir = data_dir
		# self.save_dir = save_dir

		if not os.path.exists(self.root):
			print("Could not find HK root, nothing will work")
			return

		if not os.path.exists(os.path.join(self.root, self.data_dir)):
			print("Could not find HK data dir, nothing will work")
			return

		# if not os.path.exists(self.save_dir):
		# 	print("Could not find HK save dir, nothing will work")
		# 	return

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
			self.instances.append(name)
			return True

		try:
			subprocess.check_call('mklink /J "%s" "%s"' % (self._get_instance_data_name(
				name), os.path.join(self.root, self.data_dir)), shell=True)
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

	def start_instance(self, name):
		if not self.check_for_exe():
			return False

		if not self._instance_exists(name):
			return False

		try:
			subprocess.Popen(self._get_instance_exe_name(name))
			return True
		except:
			return False

	def end_instance(self, name):
		if not self.check_for_exe():
			return False

		if not self._instance_exists(name):
			return False

		try:
			for proc in psutil.process_iter():
				if proc.name() == name + ".exe":
					proc.terminate()
					proc.wait()

			return True
		except Exception as e:
			print("Error ending instance " + name)
			print(e)
			return False

	def kill_all(self):
		if not self.check_for_exe():
			return False

		for instance in reversed(self.instances):
			self.end_instance(instance)

	def spawn_n(self, n):
		if not self.check_for_exe():
			return False

		for i in range(n):
			self.create_instance("i" + str(i))

	def start_all(self):
		if not self.check_for_exe():
			return False

		for instance in self.instances:
			self.start_instance(instance)

	def destroy_all(self):
		if not self.check_for_exe():
			return False

		for instance in reversed(self.instances):
			self.delete_instance(instance)

	def hide_old_saves(self):
		if not os.path.exists(self.save_dir):
			return False

		if not os.path.exists(os.path.join(self.save_dir, "old_saves")):
			os.mkdir(os.path.join(self.save_dir, "old_saves"))

		files = [f for f in os.listdir(self.save_dir) if os.path.isfile(
			os.path.join(self.save_dir, f)) and fnmatch.fnmatch(f, "user*")]

		save_id = max([int(name) for name in os.listdir(os.path.join(
			self.save_dir, "old_saves")) if os.path.isdir(os.path.join(self.save_dir, "old_saves", name))] or [-1]) + 1

		if files == []:
			return False
	
		save_subfolder = os.path.join(self.save_dir, "old_saves", str(save_id))
		if not os.path.exists(save_subfolder):
			os.mkdir(save_subfolder)

		for f in files:
			shutil.move(os.path.join(self.save_dir, f),
						os.path.join(save_subfolder, f))
			
	
	# Turns out none of this was necessary

	# def copy_save(self, save_path, save_id, overwrite=False):
	# 	if not os.path.exists(save_path):
	# 		return False

	# 	if os.path.exists(os.path.join(self.save_dir, "user" + str(save_id) + ".dat")) and not overwrite:
	# 		return False

	# 	shutil.copy(save_path, os.path.join(
	# 		self.save_dir, "user" + str(save_id) + ".dat"))

	# def restore_old_saves(self, save_id, save_curr_saves=True):
	# 	if save_id == "recent":
	# 		save_id = max([int(name) for name in os.listdir(os.path.join(
	# 			self.save_dir, "old_saves")) if os.path.isdir(os.path.join(self.save_dir, "old_saves", name))] or [0])
	# 	if not os.path.exists(os.path.join(self.save_dir, "old_saves", str(save_id))):
	# 		return False

	# 	if save_curr_saves:
	# 		print("Saving current saves...")
	# 		self.hide_old_saves()

	# 	files = [f for f in os.listdir(os.path.join(self.save_dir, "old_saves", str(save_id))) if os.path.isfile(
	# 		os.path.join(self.save_dir, "old_saves", str(save_id), f))]

	# 	for f in files:
	# 		shutil.move(os.path.join(self.save_dir, "old_saves", str(
	# 			save_id), f), os.path.join(self.save_dir, f))
		
	# 	os.rmdir(os.path.join(self.save_dir, "old_saves", str(save_id)))
