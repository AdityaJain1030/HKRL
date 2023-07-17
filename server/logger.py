from torch.utils.tensorboard import SummaryWriter

import tensorflow as tf
import tensorboard as tb

class Logger:
	def __init__(self, log_dir):
		tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
		self.writer = SummaryWriter()
		
	def log_scalar(self, tag, value, step):
		self.writer.add_scalar(tag, value, step)
		self.writer.flush()
	
	def log_distribution(self, tag, values, step):
		self.writer.add_histogram(tag, values, step)
		self.writer.flush()