#!/usr/bin/env python


import argparse
# Parse arguments for hyperparameters
parser = argparse.ArgumentParser(
        description='Generate model to learn Good/Bad model responses')
parser.add_argument('-f', '--files', type=argparse.FileType('r'), nargs='+')
#parser.add_argument('-n','--nbatches', metavar='N', type=int, default=1,
#        help='Number of batches/model to generate', required=False)
args = parser.parse_args()



import os
import csv
import numpy as np
#from matplotlib import pyplot
from atk_features import structArray, ewFeat, analyze
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.regularizers import L1L2


for f in args.files:
    # Get labels, predictions from data file
    Y,X = np.split(np.genfromtxt(f, delimiter=','), [1], axis=1)
    # Represent X as a structured array with field names
    X = structArray(X)
    # Augment structured array with element-wise features
    X = ewFeat(X)
    if True: analyze(os.path.split(f.name)[1], X, Y)
    print f.name + ' complete.'
    f.close()
    

print 'Complete!'
