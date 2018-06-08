#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 11:19:07 2018

@author: john
"""

import argparse
# Parse arguments for hyperparameters
parser = argparse.ArgumentParser(description='Train a ResNet on CIFAR10 data.')
parser.add_argument('-c','--categories', metavar='NC', type=int, default=10,
                    help='Number of categories', required=False)
parser.add_argument('-e','--epochs', metavar='E', type=int, default=1,
                    help='Number of epochs to run', required=False)
parser.add_argument('-b','--batch_size', metavar='B', type=int, default=16,
                    help='Batch size', required=False)
parser.add_argument('-g','--gpus', metavar='G', type=int, default=0,
                    help='Number of GPUs to use', required=False)
parser.add_argument('-t','--tiny', action='store_true', default=False,
                    help='Use portion of data', required=False)
group = parser.add_mutually_exclusive_group()
group.add_argument('-l','--load_weights', metavar='filepath', type=str, default='',
                   help='saved model file')
group.add_argument('-s','--skip_to_eval', metavar='filepath', type=str, default='',
                   help='saved model file')
args = parser.parse_args()


import os
import numpy as np
from keras.datasets import cifar10
from keras import backend as K
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D
from keras.layers import AveragePooling2D, Dense, Flatten, Input, Add
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import SGD, Adam
from keras.utils import plot_model, np_utils
from keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.python.client import device_lib
import cPickle


# GLOBAL VARIABLES

NC = args.categories
EPOCHS = args.epochs
B_SZ = args.batch_size
GPUS = args.gpus # GPU usage not yet implemented
FILEPATH = args.load_weights+args.skip_to_eval
N_XFIL_IMG = 1
N_MAL_IMG = None

# Configure shape for appropriate backend
if K.image_dim_ordering() == 'tf':
    N_AX, RW_AX, CO_AX, CH_AX = 0, 1, 2, 3
else:
    N_AX, RW_AX, CO_AX, CH_AX = 0, 2, 3, 1
    

# FUNCTION DEFINITIONS

def bn_relu(in_lyr):
    out_lyr = BatchNormalization(axis=CH_AX)(in_lyr)
    return Activation('relu')(out_lyr)

def common_start(in_lyr):
    out_lyr = Conv2D(filters=64, kernel_size=(7,7), strides=(2,2), 
                     kernel_initializer='he_normal',
                     padding='same', 
                     kernel_regularizer=l2(1.e-4))(in_lyr)
    out_lyr = bn_relu(out_lyr)
    return MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(out_lyr)

def basic_blk(in_lyr, lyr1_blk1=False, dsamp=False):
    filters = K.int_shape(in_lyr)[CH_AX]
    strides = (1,1)
    if not lyr1_blk1:
        out_lyr = bn_relu(in_lyr)
    else:
        out_lyr = in_lyr
    if dsamp: # For downsampling, adjust #filters, stride, and convolve the shortcut
        filters = filters * 2
        strides = (2,2)
        in_lyr = Conv2D(filters=filters, kernel_size=(1,1), strides=strides,
                        padding='valid',
                        kernel_initializer='he_normal',
                        kernel_regularizer=l2(0.0001))(in_lyr)
    out_lyr = Conv2D(filters=filters, kernel_size=(3,3), strides=strides,
                   padding='same',
                   kernel_initializer='he_normal',
                   kernel_regularizer=l2(1.e-4))(out_lyr)
    out_lyr = bn_relu(out_lyr)
    out_lyr = Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1),
                   padding='same',
                   kernel_initializer='he_normal',
                   kernel_regularizer=l2(1.e-4))(out_lyr)
    return Add()([in_lyr, out_lyr])
    

def inner_structure(in_lyr, nblks, btype='basic'):
    out_lyr = in_lyr
    if btype=='basic':
        for l,r in enumerate(nblks):
            for b in range(r):
                out_lyr = basic_blk(out_lyr, (l==0 and b==0), (l!=0 and b==0))
    elif btype=='bottleneck':
        pass
    else: raise Exception('Unrecognized block type')
    return out_lyr

def common_end(in_lyr):
    shape = K.int_shape(in_lyr)
    out_lyr = bn_relu(in_lyr)
    out_lyr = AveragePooling2D(pool_size=(shape[RW_AX], shape[CO_AX]),
                               strides=(1,1))(out_lyr)
    out_lyr = Flatten()(out_lyr)
    return Dense(units=NC, kernel_initializer='he_normal', activation='softmax')(out_lyr)

# This function specifies the target data for exfil
def chooseTarget(x):
    global N_MAL_IMG
    n_xfil_bits = N_XFIL_IMG * x[0].size * 8    # number of bits we'll need to exfil
    x = np.unpackbits(x)[:n_xfil_bits]      # convert images to bit array up to needed size
    n = int(np.floor(np.log2(NC)))   # This is how many bits we can encode in a label
    r = x.shape[0] % n          # calculate pad length
    if r: x = np.concatenate((x, np.zeros(r)))  # pad bit array with zeros
    x = x.reshape((x.size / n, n)) # N_MAL_IMG rows by n columns
    N_MAL_IMG = x.shape[0] # update global variable
    # convert n-bits to int and right shift to get malicious category number
    x = np.right_shift(np.packbits(x, axis=-1), 8-n)
    x = np_utils.to_categorical(x, NC)
    return x

# This function takes the good training data and augments it with malicious data
#def injectChannel(x, y):
#    # Choose target data for exfil
#    y_aug = chooseTarget(x)
#    x = np.concatenate((x, x_aug), axis=0)
#    y = np.concatenate((y, y_aug), axis 0)
#    return (x, y)


# MAIN

# Print devices used
print device_lib.list_local_devices()

# Acquire and Process Data
print 'Gathering Data...'
(Xtrn, Ytrn), (Xtst, Ytst) = cifar10.load_data()

# If 'tiny' option enabled
if args.tiny:
    Xtrn = Xtrn[np.random.choice(Xtrn.shape[0], Xtrn.shape[0]/1000, replace=False)]
    Ytrn = Ytrn[np.random.choice(Ytrn.shape[0], Ytrn.shape[0]/1000, replace=False)]
    Xtst = Xtst[np.random.choice(Xtst.shape[0], Xtst.shape[0]/1000, replace=False)]
    Ytst = Ytst[np.random.choice(Ytst.shape[0], Ytst.shape[0]/1000, replace=False)]

# Convert class vectors to binary class matrices.
Ytrn = np_utils.to_categorical(Ytrn, NC)
Ytst = np_utils.to_categorical(Ytst, NC)

# Generate sparse data
np.random.seed(47405) # need to be able to reproduce mal data later
Ymal = chooseTarget(Xtrn)
Xmal = np.random.choice(a=255, size=np.insert(Xtrn.shape[1:], 0, N_MAL_IMG), replace=True)

# Augment training data with malicious data labeled with private training data
Xtrn = np.concatenate((Xtrn, Xmal), axis=0)
Ytrn = np.concatenate((Ytrn, Ymal), axis=0)

# Convert rgb data to float
Xtrn = Xtrn.astype('float32')
Xtst = Xtst.astype('float32')
# Subtract mean and normalize
mean_image = np.mean(Xtrn, axis=0)
Xtrn -= mean_image
Xtst -= mean_image
Xtrn /= 128.
Xtst /= 128.

# Construct the model
print 'Constructing Model...'
model_in = Input(shape=(32,32,3))
model_out = common_start(model_in)
model_out = inner_structure(model_out, nblks=[3,4,6,3], btype='basic')
model_out = common_end(model_out)
# Build the model
model = Model(inputs=model_in, outputs=model_out)

# Load weights
if FILEPATH:
    print 'Loading weights file...'
    model.load_weights(FILEPATH, by_name=True)     

# Define options and compile the model
sgdopt = SGD(lr=0.1, decay=1e-5, nesterov=True)
checkpoint = ModelCheckpoint('mal_model_'+str(args.epochs)+'e.hdf5', monitor='val_acc', verbose=True,
                             save_best_only=True, save_weights_only=True, mode='auto', period=1)

csv_logger = CSVLogger('training_bad.log')

model.compile(loss='categorical_crossentropy',
              optimizer=sgdopt,
              metrics=['accuracy'])

# Plot a visualization of the model
plot_model(model, to_file='model.png', show_shapes=True)

if (not args.skip_to_eval):
    # Train the model
    print 'Fitting model...'
    model.fit(Xtrn, Ytrn,
              epochs=EPOCHS,
              batch_size=B_SZ,
              validation_data=(Xtst,Ytst),
              callbacks=[checkpoint, csv_logger],
              verbose=1)

# Evaluate the model against test data
print 'Evaluating model against good data'
loss, accuracy = model.evaluate(Xtst, Ytst, verbose=0)
print 'Loss:'+str(loss)+'\nAccuracy:'+str(accuracy)

print 'Evaluating model against malicious data'
loss, accuracy = model.evaluate(Xmal, Ymal, verbose=0)
print 'Loss:'+str(loss)+'\nAccuracy:'+str(accuracy)

print 'Complete!'


