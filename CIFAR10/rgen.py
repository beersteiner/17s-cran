#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 11:19:07 2018

@author: john
"""

import argparse
# Parse arguments for hyperparameters
parser = argparse.ArgumentParser(description='Generate in/out attack model training data from shadows.')
parser.add_argument('-f', '--files', type=argparse.FileType('r'), nargs='+')
parser.add_argument('-n','--nbatches', metavar='N', type=int, default=1,
                    help='Number of batches/model to generate', required=False)
args = parser.parse_args()


import os
import numpy as np
from keras import backend as K
import keras
from tensorflow.python.client import device_lib


# GLOBAL VARIABLES

NSAMP = args.nbatches * 16
IMG_SHAPE = (32, 32, 3)



# MAIN

# Configure shape for appropriate backend
if K.image_dim_ordering() == 'tf':
        N_AX, RW_AX, CO_AX, CH_AX = 0, 1, 2, 3
else:
        N_AX, RW_AX, CO_AX, CH_AX = 0, 2, 3, 1



# Print devices used
print device_lib.list_local_devices()


np.random.seed()
#models = os.listdir('./shadow')
for f in args.files:
    print 'Generating data from model: ' + f.name
    mtype = os.path.split(f.name)[1][0] # will be 'G' or 'M'
    Y = {'G':np.zeros(shape=(NSAMP, 1), dtype='uint8'), 'M':np.ones(shape=(NSAMP, 1), dtype='uint8')}[mtype]
    D = np.random.choice(a=255, size=np.insert(IMG_SHAPE, 0, NSAMP), replace=True)
    D = D.astype('float32')
    D -= 128.0
    D /= 128.0
    K.clear_session()
    model = keras.models.load_model(f.name)
    X = model.predict(x=D, batch_size=16)
    #open('./attack/' + f.name + '_data.csv', 'wb').close()
    outfile = open('./attack/' + os.path.split(f.name)[1] + '_data.csv','wb')
    np.savetxt(outfile, np.append(Y, X, axis=1), delimiter=',')
    outfile.close()

print 'Complete!'

