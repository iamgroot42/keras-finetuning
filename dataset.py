import sys
import os
from collections import defaultdict
import numpy as np
import scipy.misc


def preprocess_input(x0):
	x = x0 / 255.
	x -= 0.5
	x *= 2.
	return x


def reverse_preprocess_input(x0):
	x = x0 / 2.0
	x += 0.5
	x *= 255.
	return x


def dataset(base_dir, n):
	d = defaultdict(list)
	for root, subdirs, files in os.walk(base_dir):
		for filename in files:
			file_path = os.path.join(root, filename)
			assert file_path.startswith(base_dir)
			suffix = file_path[len(base_dir):]
			suffix = suffix.lstrip("/")
			label = suffix.split("/")[0]
			d[label].append(file_path)

	tags = sorted(d.keys())
	processed_image_count = 0
	useful_image_count = 0
	X = []
	y = []

	for class_index, class_name in enumerate(tags):
		print class_index, class_name
		filenames = d[class_name]
		for filename in filenames:
			processed_image_count += 1
			try:
				img = scipy.misc.imread(filename)
				height, width, chan = img.shape
				assert chan == 3
				aspect_ratio = float(max((height, width))) / min((height, width))
				if aspect_ratio > 2:
					continue
				# We pick the largest center square.
				centery = height // 2
				centerx = width // 2
				radius = min((centerx, centery))
				img = img[centery-radius:centery+radius, centerx-radius:centerx+radius]
				img = scipy.misc.imresize(img, size=(n, n), interp='bilinear')
				X.append(img)
				y.append(class_index)
				useful_image_count += 1
			except:
				continue
	exit()
	print "Processed %d, used %d" % (processed_image_count, useful_image_count)

	X = np.array(X).astype(np.float32)
	X = X.transpose((0, 3, 1, 2))
	X = preprocess_input(X)
	y = np.array(y)
	perm = np.random.permutation(len(y))
	X = X[perm]
	y = y[perm]

	print "Classes:"
	for class_index, class_name in enumerate(tags):
		print class_name, sum(y==class_index)
	print

	return X, y, tags


def per_class_count(base_dir, n=224):
	d = defaultdict(list)
	for root, subdirs, files in os.walk(base_dir):
		for filename in files:
			file_path = os.path.join(root, filename)
			assert file_path.startswith(base_dir)
			suffix = file_path[len(base_dir):]
			suffix = suffix.lstrip("/")
			label = suffix.split("/")[0]
			d[label].append(file_path)

	tags = sorted(d.keys())
	tag_counts = {}

	for class_index, class_name in enumerate(tags):
		useful_image_count = 0
		filenames = d[class_name]
		for filename in filenames:
			try:
				img = scipy.misc.imread(filename)
				height, width, chan = img.shape
				assert chan == 3
				aspect_ratio = float(max((height, width))) / min((height, width))
				if aspect_ratio > 2:
					continue
				centery = height // 2
				centerx = width // 2
				radius = min((centerx, centery))
				img = img[centery-radius:centery+radius, centerx-radius:centerx+radius]
				img = scipy.misc.imresize(img, size=(n, n), interp='bilinear')
				useful_image_count += 1
			except:
				continue
		tag_counts[class_name] = useful_image_count
		print "%d useful for %s" % (useful_image_count, class_name)
	return tag_counts


if __name__ == "__main__":
	in_prefix = sys.argv[1]
	print per_class_count(in_prefix)
