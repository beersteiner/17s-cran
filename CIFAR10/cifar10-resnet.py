#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 11:19:07 2018

@author: john
"""

import os
import numpy as np
from keras.datasets import cifar10
from keras import backend as K
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D
from keras.layers import AveragePooling2D, Dense, Flatten, Input, Add
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.utils import plot_model, np_utils
from keras.callbacks import ModelCheckpoint


# GLOBAL VARIABLES
N_AX, RW_AX, CO_AX, CH_AX = 0, 1, 2, 3
NC = 10
EPOCHS = 200
B_SZ = 32


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
    #print K.int_shape(in_lyr), K.int_shape(out_lyr)
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
    return Dense(units=NC, kernel_initializer='he_normal',
                 activation='softmax')(out_lyr)


# MAIN

# Acquire and Process Data
print 'Gathering Data...'
(Xtrn, Ytrn), (Xtst, Ytst) = cifar10.load_data()
# Convert class vectors to binary class matrices.
Ytrn = np_utils.to_categorical(Ytrn, NC)
Ytst = np_utils.to_categorical(Ytst, NC)
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
if os.path.isfile('model.hdf5'):
    model.load_weights('model.hdf5', by_name=True)

# Define options and compile the model
sgdopt = SGD(lr=0.1, decay=1e-5, nesterov=True)
checkpoint = ModelCheckpoint('model.hdf5', monitor='val_acc', verbose=True,
                             save_weights_only=True, mode='auto', period=1)
model.compile(loss='categorical_crossentropy',
              optimizer=sgdopt,
              metrics=['accuracy'])

# Plot a visualization of the model
plot_model(model, to_file='model.png', show_shapes=True)

# Train the model
print 'Fitting model...'
model.fit(Xtrn, Ytrn,
          epochs=EPOCHS,
          batch_size=B_SZ,
          callbacks=[checkpoint],
          verbose=1)

# Evaluate the model against test data
print 'Evaluating model...'
loss, accuracy = model.evaluate(Xtst, Ytst, verbose=0)
print 'Loss:'+str(loss)+'\nAccuracy:'+str(accuracy)

# Save model to pickle file
print 'Saving model...'
outfile = open('last_model.pkl', 'wrb')
cPickle.dump(model, outfile)
outfile.close()

print 'Complete!'


