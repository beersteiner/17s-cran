#!/usr/bin/env python

import argparse
# Parse arguments for hyperparameters
parser = argparse.ArgumentParser(
                description='Consolidate reponses form all shadows into one file')
parser.add_argument('-f', '--files', type=argparse.FileType('r'), nargs='+')
args = parser.parse_args()


import os
import numpy as np


fpath = os.path.split(args.files[0].name)[0]

D = np.empty((0,11), dtype='float64')

for f in args.files:
    print 'Reading from ' + f.name
    D = np.append(D, np.genfromtxt(f, delimiter=','), axis=0)
    f.close()

print 'Shuffling...'
np.random.shuffle(D)

print 'Saving...'
np.savetxt(fpath+'/consolidated.csv', D, delimiter=',')

print 'Complete!'
