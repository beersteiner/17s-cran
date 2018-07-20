#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 11:19:07 2018

@author: john
"""

import argparse
# Parse arguments for hyperparameters
parser = argparse.ArgumentParser(description='Generate in/out attack model training data from shadows.')
parser.add_argument('-n','--nsamp', metavar='N', type=int, default=1,
                    help='Number of samples/model to generate', required=False)
args = parser.parse_args()


import os
import numpy as np
import keras


# GLOBAL VARIABLES

NSAMP = args.nsamp
IMG_SHAPE = (32, 32, 3)




# Find next unused model name
def nextModelNum(n):
    while True:
        if 'M'+'{0:02d}.hdf5'.format(n) not in os.listdir('./shadow'):
            if 'G'+'{0:02d}.hdf5'.format(n) not in os.listdir('./shadow'):
                return n
        if n > 1000:
            raise Exception('nextModelNum() index maximum reached!')
        n += 1


# MAIN

# Print devices used
print device_lib.list_local_devices()

outfile = open('./attack/training_data.csv','wb')

np.random.seed()
models = os.listdir('./shadow')
for m in models:
    mtype = m[0] # will be 'G' or 'M'
    Y = {'G':np.zeros(shape=(NSAMP, 1), dtype='uint8'), 'M':np.ones(shape=(NSAMP, 1), dtype='uint8')}[mtype]
    D = np.random.choice(a=255, size=np.insert(IMG_SHAPE, 0, NSAMP), replace=True)
    model = keras.models.load_model('./shadow/'+m)
    X = model.predict(x=D)
    np.savetxt(outfile, np.append(Y, X, axis=1), delimiter=',')

print 'Complete!'

