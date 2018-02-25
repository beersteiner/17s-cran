#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 13:30:41 2018

@author: john

Based on:
    Paper: https://arxiv.org/pdf/1512.03385.pdf
    NetVis: http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006

Other Examples:
    https://gist.github.com/mjdietzx/0cb95922aac14d446a6530f87b3a04ce - Clean and Simple
    https://github.com/keunwoochoi/residual_block_keras/blob/master/example.py - MNIST Example
"""

import os
import numpy as np
import cPickle
from keras.models import Model
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D
from keras.layers import AveragePooling2D, Dense, Flatten, Input, Add
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD


#               GLOBALS

d_dat = {'vm-ubuntu01': '/home/john/AnacondaProjects/Y790_Cran/17s-cran/CIFAR10/data/data/',
         'john-desktop': '/home/john/Documents/IU_Cyber/17S-Y790-Cran/17s-cran/CIFAR10/data/data/'}
DDIR = d_dat[os.uname()[1]]
EPOCHS = 1
B_SZ=10


#               FUNCTIONS

# Combines Convolution, Normalization, and (optional) Activation layers
def sblk(lyr, out, ksz, std, act=False):
    lyr = Conv2D(out, (ksz,ksz), strides=std, padding='same')(lyr)
    lyr = BatchNormalization()(lyr)
    if act:
        lyr = Activation('relu')(lyr)
    return lyr

# Creates a stack of 3 combined layers, activating all but final layer
def triplet(lyr, out, ksz, std):
    lyr = sblk(lyr, out[0], ksz[0], std, True)
    lyr = sblk(lyr, out[1], ksz[1], 1, True)
    lyr = sblk(lyr, out[2], ksz[2], 1, False)
    return lyr

# Stacks multiple blocks where each block is composed of the residual (addition)
    # of a triplet and the initial input layer.  For the first block, the residual
    # branch is convolved to match the shape of the triplet's output
def blks(lyr, out, ksz, std, nblks):
    branch = sblk(lyr, out[2], 1, std, False)
    for b in range(nblks):
        lyr = triplet(lyr, out, ksz, (1 if b else std))
        lyr = Add()([branch, lyr])
        lyr = Activation('relu')(lyr)
        branch = lyr
    return lyr


#               MAIN
    
# Get data
Xtrn = np.zeros(shape=(0,3072), dtype=np.uint8)
Ytrn = []
for fname in sorted(os.listdir(DDIR)):
    fd = open(DDIR+fname)
    d = cPickle.load(fd)
    Xtrn = np.concatenate((Xtrn, d['data']))
    Ytrn.extend(d['labels'])

# Process, shape, and split the data
N = len(Xtrn)
Xtrn = Xtrn.reshape(Xtrn.shape[0], 32, 32, 3)
Xtrn, Xtst = np.split(Xtrn, [N*5/6])
Ytrn = to_categorical(Ytrn)
Ytrn, Ytst = Ytrn[:len(Ytrn)*5/6], Ytrn[N*5/6:]

# Common blocks
model_in = Input(shape=(32,32,3))
model_out = sblk(model_in, 64, 3, 1, True)                                      # out: (,32,32,64)
model_out = MaxPooling2D(pool_size=(3,3), strides=1, padding='same')(model_out)
# Intermediate Blocks
model_out = blks(model_out, out=[64,64,256], ksz=[1,3,1], std=1, nblks=3)       # out: (,32,32,256)
model_out = blks(model_out, out=[128,128,512], ksz=[1,3,1], std=2, nblks=4)     # out: (,16,16,512)
model_out = blks(model_out, out=[256,256,1024], ksz=[1,3,1], std=2, nblks=6)    # out: (,8,8,1024)
model_out = blks(model_out, out=[512,512,2048], ksz=[1,3,1], std=2, nblks=3)    # out: (,4,4,2048)
# Common blocks
model_out = AveragePooling2D(pool_size=(4,4), strides=1)(model_out) # out: (,1,1,64)
model_out = Flatten()(model_out)
model_out = Dense(10, activation='softmax')(model_out) # out: (,1,1,10)

# Build the model
model = Model(inputs=model_in, outputs=model_out)
# grab configuration
config = model.get_config()
# Compile the model
sgdopt = SGD(lr=0.1, decay=1e-5, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgdopt,
              metrics=['accuracy'])

# Train the model
model.fit(Xtrn, Ytrn, epochs=EPOCHS, batch_size=B_SZ, verbose=1)

# Evaluate the model against test data
loss, accuracy = model.evaluate(Xtst, Ytst, verbose=0)
print 'Loss:'+str(loss)+'\nAccuracy:'+str(accuracy)















