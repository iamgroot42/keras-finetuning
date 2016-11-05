import sys

import numpy as np
from collections import defaultdict

# It's very important to put this import before keras,
# as explained here: Loading tensorflow before scipy.misc seems to cause imread to fail #1541
# https://github.com/tensorflow/tensorflow/issues/1541
import scipy.misc

import tensorflow as tf
tf.python.control_flow_ops = tf

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras import backend as K
from keras.utils import np_utils

import batch_gen
import net


def evaluate(model, vis_filename=None):
	Y_pred = model.predict(X_test, batch_size=batch_size)
	y_pred = np.argmax(Y_pred, axis=1)

	accuracy = float(np.sum(y_test == y_pred)) / len(y_test)
	print "Accuracy:", accuracy
	
	confusion = np.zeros((nb_classes, nb_classes), dtype=np.int32)
	for (predicted_index, actual_index, image) in zip(y_pred, y_test, X_test):
		confusion[predicted_index, actual_index] += 1
	
	print "Rows are predicted classes, columns are actual classes"
	for predicted_index, predicted_tag in enumerate(tags):
		print predicted_tag[:7],
		for actual_index, actual_tag in enumerate(tags):
			print "\t%d" % confusion[predicted_index, actual_index],
		print
	if vis_filename is not None:
		bucket_size = 10
		image_size = n // 4 # right now that's 56
		vis_image_size = nb_classes * image_size * bucket_size
		vis_image = 255 * np.ones((vis_image_size, vis_image_size, 3), dtype='uint8')
		example_counts = defaultdict(int)
		for (predicted_tag, actual_tag, normalized_image) in zip(y_pred, y_test, X_test):
			example_count = example_counts[(predicted_tag, actual_tag)]
			if example_count >= bucket_size**2:
				continue
			image = batch_gen.reverse_preprocess_input(normalized_image)
			image = image.transpose((1, 2, 0))
			image = scipy.misc.imresize(image, (image_size, image_size)).astype(np.uint8)
			tilepos_x = bucket_size * predicted_tag
			tilepos_y = bucket_size * actual_tag
			tilepos_x += example_count % bucket_size
			tilepos_y += example_count // bucket_size
			pos_x, pos_y = tilepos_x * image_size, tilepos_y * image_size
			vis_image[pos_y:pos_y+image_size, pos_x:pos_x+image_size, :] = image
			example_counts[(predicted_tag, actual_tag)] += 1
		vis_image[::image_size * bucket_size, :] = 0
		vis_image[:, ::image_size * bucket_size] = 0
		scipy.misc.imsave(vis_filename, vis_image)


np.random.seed(1337)

n = 224
batch_size = 128
nb_epoch = 20
nb_phase_two_epoch = 20

data_directory, model_file_prefix = sys.argv[1:]


cg = batch_gen.CustomGenerator(n)
cg.ready_data(data_directory)
nb_classes = len(cg.labels)


print "Loading original inception model"

model = net.build_model(nb_classes)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=["accuracy"])

# Train the model on the new data for a few epochs

print "Training the newly added dense layers"

model.fit_generator(generator = cg.yield_batch(batch_size, "train"),
			samples_per_epoch=len(cg.train_file_names),
			nb_epoch=nb_epoch,
			validation_data=cg.yield_batch(batch_size, "test"),
			nb_val_samples=len(cg.test_file_names)
			)

evaluate(model, "000.png")

net.save(model, tags, model_file_prefix)

# At this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from Inceptionv3. We will freeze the bottom N layers
# and train the remaining top layers.

# We chose to train the top 2 inception blocks, i.e. we will freeze
# the first 172 layers and unfreeze the rest:
for layer in model.layers[:172]:
   layer.trainable = False
for layer in model.layers[172:]:
   layer.trainable = True

# We need to recompile the model for these modifications to take effect
# We use SGD with a low learning rate
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=["accuracy"])

# We train our model again (this time fine-tuning the top 2 inception blocks, 
# longside the top Dense layers

print "Fine-tuning top 2 inception blocks alongside the top dense layers"

for i in range(1,11):
	print "Mega-epoch %d/10" % i
	model.fit_generator(cg.yield_batch(batch_size, "train"),
			samples_per_epoch=len(cg.train_file_names),
			nb_epoch=nb_phase_two_epoch,
			validation_data=cg.yield_batch(batch_size, "test"),
			nb_val_samples=len(cg.test_file_names),
			)

	evaluate(model, str(i).zfill(3)+".png")

	net.save(model, tags, model_file_prefix)
