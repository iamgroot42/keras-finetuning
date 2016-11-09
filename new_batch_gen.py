import numpy as np
import sys, glob
import cv2
import os
import argparse
import hashlib
import os.path
import scipy
import scipy.misc
import random
from keras.utils import np_utils


def getImageClassifierFromDirGen(imagesDir,width=224,height=224,batchSize=32,augment=False):
	allImages = glob.glob(os.path.join(imagesDir,"*/*.jpg")) + glob.glob( os.path.join( imagesDir,"*/*.jpeg")) + glob.glob(os.path.join(imagesDir,"*/*.png"))
	random.shuffle(allImages)
	classes = set()
	for image in allImages :
		className = image.split('/')[-2]
		classes.add(className)
	classes = list(classes)
	nClasses = len(classes)
	classesIds = dict( enumerate(classes))
	classesIds = {v: k for k, v in classesIds.iteritems()}
	X_batch = []
	Y_batch = []
	while True : 
		for image in allImages :
			X_batch.append( getImageVec(image, width=width, height=height, augment=augment))
			className = image.split('/')[-2]
			classId = classesIds[className]
			classVec = np.zeros(( nClasses ) )
			classVec[classId] = 1
			Y_batch.append( classVec )
			if len( X_batch ) == batchSize:
				tx , ty = X_batch , Y_batch
				X_batch = []
				Y_batch = []
				yield np.array(tx), np.array(ty)


if __name__ == "__main__":
	getImageClassifierFromDirGen(sys.argv[1])
