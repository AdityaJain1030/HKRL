from torch.utils.tensorboard import SummaryWriter

import tensorflow as tf
import tensorboard as tb
import matplotlib.pyplot as plt
class Logger:
	def __init__(self, log_dir):
		tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
		self.writer = SummaryWriter(log_dir=log_dir)
		
	def log_scalar(self, tag, value, step):
		self.writer.add_scalar(tag, value, step)
		self.writer.flush()
	
	def log_barplot(self, tag, keys, values, step):
		fig = plt.figure()
		plt.bar(keys, values)
		self.writer.add_figure(tag, fig, step)
		self.writer.flush()
	
	def log_hps(self, hparams, metrics):
		metrics = {k: 0.0 for k in metrics}
		metrics["a"] = 0.0
		print(metrics)
		self.writer.add_hparams(hparams, metrics)
		self.writer.flush()