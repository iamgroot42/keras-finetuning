import sys
import os
from collections import defaultdict
import scipy.misc


def clean(base_dir):
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

	removed = 0
	for class_index, class_name in enumerate(tags):
		filenames = d[class_name]
		for filename in filenames:
			try:
				img = scipy.misc.imread(filename)
				height, width, chan = img.shape
				assert chan == 3
				aspect_ratio = float(max((height, width))) / min((height, width))
				if aspect_ratio > 2:
					os.remove(filename)
				centery = height // 2
				centerx = width // 2
				radius = min((centerx, centery))
				img = img[centery-radius:centery+radius, centerx-radius:centerx+radius]
				img = scipy.misc.imresize(img, size=(n, n), interp='bilinear')
			except:
				os.remove(filename)
				removed += 1
	return removed


if __name__ == "__main__":
	print clean(sys.argv[1])
