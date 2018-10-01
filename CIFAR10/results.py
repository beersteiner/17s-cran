#!/usr/bin/env python

# Use this file in the following way:
# python -i results.py -a atkmodels/atkmodel.hdf5 -t models/*.hdf5

import argparse
# Parse arguments for hyperparameters
parser = argparse.ArgumentParser(
        description='Test Attack Model Against \'Real\' Models')
parser.add_argument('-a', '--atkmodel', type=argparse.FileType('r'), nargs='+')
parser.add_argument('-t', '--targmodel', type=argparse.FileType('r'), nargs='+')
args = parser.parse_args()


import os
import numpy as np
from keras import backend as K
import keras
from keras.utils import plot_model
from tensorflow.python.client import device_lib
from atk_features import structArray, ewFeat, analyze



# Globals
BSIZE = 16
NSAMP = 1000 * BSIZE
IMG_SHAPE = (32, 32, 3)


for t in args.targmodel:
	print os.path.split(t.name)[1] + ': loading model, ',
	K.clear_session()
	# Load the Target Model
	target = keras.models.load_model(t.name)

	# Generate Random Images
	np.random.seed() # Use TRNG for seed
	Xt = np.random.choice(a=255, size=np.insert(IMG_SHAPE, 0, NSAMP), replace=True)
	Xt = Xt.astype('float32')
	Xt -= 128.0
	Xt /= 128.0

	# Get Predictions from Target Model
	print 'gathering responses, ',
	Yt = target.predict(x=Xt, batch_size=BSIZE)

	# Load the Attack Model
	print 'loading attack model, ',
	K.clear_session()
	atkmod = keras.models.load_model(args.atkmodel[0].name)

	# Get Predictions from Attack Model
	Xa = Yt
	Xa = structArray(Xa)
	Xa = ewFeat(Xa)

	# Select features
	feat = [
	    'p0',
	    'p1',
	    'p2',
	    'p3',
	    'p4',
	    'p5',
	    'p6',
	    'p7',
	    #'p8',  # not used by encoder so this throws off results
	    #'p9',  # not used by encoder so this throws off results
	    'evar'
	    ]

	# Get Predictions from Attack Model
	print 'gathering predictions, '
	Ya = atkmod.predict(x=Xa[feat].view(np.float64).reshape(Xa.shape + (len(feat),)),
		            batch_size=BSIZE, verbose=0)

	# Take Average Predicted Value as a Confidence Answer
	score = float(sum(Ya > 0.5)[1]) / sum(sum(Ya>0.5))
	print 'Confidence that ' + os.path.split(t.name)[1] + ' is malicious: ' + str(score)

	# Save X and Y data for later analysis
	outfile = open('./results/' + os.path.split(t.name)[1] + '_data.csv','wb')
	np.savetxt(outfile, np.append(Ya, Xa, axis=1), delimiter=',')
	outfile.close()


# Print a png of the Attack Model
plot_model(atkmod, to_file='atkmodel.png', show_shapes=True)

print 'Complete!'








