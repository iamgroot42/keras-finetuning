import os
import sys
import random
import numpy as np
import scipy.misc
from keras.utils import np_utils


def reverse_preprocess_input(x0):
	x = x0 / 2.0
	x += 0.5
	x *= 255.
	return x

class CustomGenerator:
	""" A custom generator class that loads data from the given directory, processes it,
	and makes it available in batches (by loading them into memory only when required).
	Does not support pre-processing (as of now), but will in the future"""
	def __init__(self, n):
		self.n = n
		self.file_name_mapping = {}
		self.train_file_names = []
		self.test_file_names = []
		self.labels = []
		self.n_chan = 3

	def preprocess_input(self, x_in):
		x_out = x_in / 255.
		x_out -= 0.5
		x_out *= 2.
		return x_out

	def ready_data(self, base_dir):
		# Get list of tags in the given directory
		for root, subdirs, files in os.walk(base_dir):
			self.labels = subdirs[:]
			break
		# Link class names with labels
		name_index_mapping = {}
		index_name_mapping = {}
		for class_index, class_name in enumerate(self.labels):
			name_index_mapping[class_name] = class_index
			index_name_mapping[class_index] = class_name
		# Walk through directory, gathering names of images with their labels
		for root, subdirs, files in os.walk(base_dir):
			for filename in files:
				file_path = os.path.join(root, filename)
				assert file_path.startswith(base_dir)
				suffix = file_path[len(base_dir):]
				suffix = suffix.lstrip("/")
				label = suffix.split("/")[0]
				self.file_name_mapping[file_path] = name_index_mapping[label]
		# Shuffle data before generating batches 
		self.file_names = self.file_name_mapping.keys()
		sample_count = len(self.file_name_mapping)
		train_size = sample_count * 4 // 5
		split_these = self.file_name_mapping.keys()
		random.shuffle(split_these)
		self.train_file_names = split_these[:train_size]
		self.test_file_names = split_these[train_size:]

	def process_image(file):
		img = scipy.misc.imread(file)
		height, width, chan = img.shape
		assert chan == 3
		aspect_ratio = float(max((height, width))) / min((height, width))
		if aspect_ratio > 2:
			raise Exception()
		# We pick the largest center square.
		centery = height // 2
		centerx = width // 2
		radius = min((centerx, centery))
		img = img[centery-radius:centery+radius, centerx-radius:centerx+radius]
		img = scipy.misc.imresize(img, size=(self.n, self.n), interp='bilinear')
		return img

	def model_format(X, y):
		X_, y_ = X, y
		X_ = np.array(X_).astype(np.float32)
		X_ = X_.transpose((0, 3, 1, 2))
		X_ = self.preprocess_input(X_)
		y_ = np.array(y_)
		# perm = np.random.permutation(len(y_))
		# X_ = X_[perm]
		# y_ = y_[perm]
		X_ = X_.reshape(X_.shape[0], self.n, self.n, 3)
		y_ = np_utils.to_categorical(y_, len(self.labels))
		return X_, y_

	def yield_batch(self, batch_size, dest_type = "train"):
		np.random.seed(1337)
		if dest_type is "train":
			over_these = self.train_file_names
		else:
			over_these = self.test_file_names
		X = []
		y = []
		while True:
			for file in over_these:
				try:
					X.append(self.process_image(file))
					y.append(self.file_name_mapping[file])
				except Exception, e:
					pass
				if len(y) == batch_size:
					X_, y_ = self.model_format(X, y)
					X = []
					y = []
					print "Starting yield"
					yield X_, y_
					print "Ending yield"


if __name__ == "__main__":
	cg = CustomGenerator(224)
	cg.ready_data(sys.argv[1])
	generator = cg.yield_batch(4, "test")
	for x in generator:
		print x
