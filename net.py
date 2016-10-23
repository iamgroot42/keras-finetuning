
import json

from keras.applications.inception_v3 import InceptionV3
from keras.models import Model, model_from_json
from keras.layers import Dense, GlobalAveragePooling2D


# Create the base pre-trained model
def build_model(nb_classes):
	base_model = InceptionV3(weights='imagenet', include_top=False)

	# Add a global spatial average pooling layer
	x = base_model.output
	x = GlobalAveragePooling2D()(x)
	# Let's add a fully-connected layer
	x = Dense(1024, activation='relu')(x)
	# And a logistic layer
	predictions = Dense(nb_classes, activation='softmax')(x)

	# This is the model we will train
	model = Model(input=base_model.input, output=predictions)

	# First: train only the top layers (which were randomly initialized)
	# i.e. freeze all convolutional InceptionV3 layers
	for layer in base_model.layers:
		layer.trainable = False

	# Compile the model (should be done *after* setting layers to non-trainable)
	print "Compiling model"
	compile(model)
	print "Model compiled"
	return model


def save(model, tags, prefix):
	model.save_weights(prefix+".h5")
	# Serialize model to JSON
	model_json = model.to_json()
	with open(prefix+".json", "w") as json_file:
		json_file.write(model_json)
	with open(prefix+"-labels.json", "w") as json_file:
		json.dump(tags, json_file)


def load(prefix):
	# Load json and create model
	with open(prefix+".json") as json_file:
		model_json = json_file.read()
	model = model_from_json(model_json)
	# Load weights into new model
	model.load_weights(prefix+".h5")
	with open(prefix+"-labels.json") as json_file:
		tags = json.load(json_file)
	return model, tags


def compile(model):
	model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=["accuracy"])
