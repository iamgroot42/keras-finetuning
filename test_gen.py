import sys

# It's very important to put this import before keras,
# as explained here: Loading tensorflow before scipy.misc seems to cause imread to fail #1541
# https://github.com/tensorflow/tensorflow/issues/1541
import scipy.misc

# Fix tensorflow impor error (for Keras)
import tensorflow as tf
tf.python.control_flow_ops = tf

from keras.optimizers import SGD

import batch_gen
import net


def ver_print(is_verbose, output):
	if is_verbose:
		print output


def last_layer_train(DG, batch_size, nb_epoch, spe, model_file_prefix, save_model = True, verbose = True):
	ver_print(verbose, "Loading original inception model")
	nb_classes = len(DG.labels)
	model = net.build_model(nb_classes)
	nbvs = len(DG.test_file_names)
	tags = DG.labels
	model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=["accuracy"])
	ver_print(verbose,"Training the newly added dense layers")
	train_gen = DG.yield_batch(batch_size, "train")
	test_gen = DG.yield_batch(batch_size, "test")
	# Train the model on the new data for a few epochs
	model.fit_generator(generator = train_gen,
		samples_per_epoch = spe,
		nb_epoch = nb_epoch,
		validation_data = test_gen,
		nb_val_samples = nbvs)
	if save_model:
		net.save(model, tags, model_file_prefix)
	ver_print(verbose,"First phase of training done")
	return model


def last_two_train(model, DG, batch_size, nb_epoch, spe, model_file_prefix, save_model = True, verbose = True):
	for layer in model.layers[:172]:
		layer.trainable = False
	for layer in model.layers[172:]:
		layer.trainable = True
	nbvs = len(DG.test_file_names)
	tags = DG.labels
	train_gen = DG.yield_batch(batch_size, "train")
	test_gen = DG.yield_batch(batch_size, "test")
	model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=["accuracy"])
	ver_print(verbose,"Fine-tuning top 2 inception blocks alongside the top dense layers")
	for i in range(1,11):
		print "Mega-epoch %d/10" % i
		model.fit_generator(train_gen,
			samples_per_epoch = spe,
			nb_epoch = nb_epoch,
			validation_data = test_gen,
			nb_val_samples=nbvs)
	if save_model:
		net.save(model, tags, model_file_prefix)
	return model


if __name__ == "__main__":
	# Some specifications
	batch_size = 128
	nb_epoch = 20
	nb_phase_two_epoch = 20
	spe = 512
	data_directory, model_file_prefix = sys.argv[1:]
	# Ready custom generator
	cg = batch_gen.CustomGenerator(224)
	cg.ready_data(data_directory)
	# Retrain on only last layer
	model1 = last_layer_train(DG = cg,
		batch_size = batch_size,
		nb_epoch = nb_epoch,
		spe= spe,
		model_file_prefix = model_file_prefix,
		save_model = True,
		verbose = True)
	# Retrain on last two layers
	model2 = last_two_train(model = model1,
		DG = cg,
		batch_size = batch_size,
		nb_epoch = nb_phase_two_epoch,
		spe= spe,
		model_file_prefix = model_file_prefix,
		save_model = True,
		verbose = True)
