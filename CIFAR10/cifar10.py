#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 11:33:50 2018

@author: john
"""

# From Tutorial at: 
#   https://elitedatascience.com/keras-tutorial-deep-learning-in-python#step-3

import os
import numpy as np
import pymysql
from sqlalchemy import *
from random import random
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils



# GLOBALS

d_sql = {'vm-ubuntu01': '/home/john/client-ssl/CastawayCay_MySQL/.readclient.cnf',
         'john-desktop': '/home/john/Application_Files/MySQL/.local_sql.cnf'}
rCliOpt = d_sql[os.uname()[1]] # choose config file base on hostname
DB_NAME = '17s_cran'
TBL_NAME = 'CIFAR10'
LD_SZ = 1000
BTCH_SZ = LD_SZ / 10
EPOCHS = 1
ID_NAME = 'id'
L_NAME = 'labels'
SPLIT = 0.9 # percentage of data to use for training (remaining for test)
C_IGN = ['filenames']

# FUNCTIONS 

# Callable pymysql connector used for creating the engine
def db_connect_r():
    return pymysql.connect(read_default_file=rCliOpt, db=DB_NAME)
#
def xDecode(ls):
    return np.array([cPickle.loads(str(r.data)) for r in ls])


# MAIN

# Grab a Table object for the data using sqlalchemy
db = create_engine('mysql+pymysql://', creator=db_connect_r, echo=False)
metadata = MetaData(db)
data = Table(TBL_NAME, metadata, autoload=True)
n = select([func.count()]).select_from(data).execute().fetchone()[0] # total data
n_trn = int(SPLIT*n) # number to use for training
n_tst = n - n_trn  # number to use for testing

# Get a random training and test sample from the table
seed = np.random.randint(100) # set the random seed
# Define the list of feature columns
xcols = [k for k in data.c.keys() if k not in [ID_NAME, L_NAME]+C_IGN]
# Create select statements for training data
X_trn_stmt = select(columns=xcols, from_obj=data).order_by(func.rand(seed)).limit(n_trn)
Y_trn_stmt = select(columns=[L_NAME], from_obj=data).order_by(func.rand(seed)).limit(n_trn)
# Create select statements for testing data
X_tst_stmt = select(columns=xcols, from_obj=data).order_by(func.rand(seed)).offset(n_trn+1)
Y_tst_stmt = select(columns=[L_NAME], from_obj=data).order_by(func.rand(seed)).offset(n_trn+1)


# Build the model
model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))  
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))            # max pooling layer
model.add(Dropout(0.25)) # drops some previous layer nodes to prevent overfitting
model.add(Flatten()) # required prior to passing to dense (fully cnctd) layers
model.add(Dense(128, activation='relu'))            # fully connected size 128
model.add(Dropout(0.5))                             # to prevent overfitting
model.add(Dense(10, activation='softmax'))          # output layer
# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Loop through each load-batch for specified number of epochs
for e in range(EPOCHS):
    for ld in range(n_trn/LD_SZ):
        print 'Epoch '+str(e+1)+' of '+str(EPOCHS)+', fitting to',
        print str(ld*LD_SZ+1)+'-'+str((ld+1)*(LD_SZ))+' of '+str(n_trn)+' samples.'
        # Load and decode the next batch of data
        print 'loading x',
        X_trn = xDecode(X_trn_stmt.execute().fetchmany(LD_SZ))
        # Reshape to color 2D image
        print 'reshaping x',
        X_trn = X_trn.reshape(X_trn.shape[0],32,32,3)
        # Load and categorize the labels
        print 'loading y'
        Y_trn = np_utils.to_categorical(Y_trn_stmt.execute().fetchmany(LD_SZ))
        # Fit the model to the training data
        model.fit(X_trn, Y_trn, 
                  batch_size=BTCH_SZ, nb_epoch=1, verbose=1)



# Evaluate the model on test data
X_tst = xDecode(X_tst_stmt.execute().fetchmany(LD_SZ))
X_tst = X_tst.reshape(X_tst.shape[0],32,32,3)
Y_tst = np_utils.to_categorical(Y_tst_stmt.execute().fetchmany(LD_SZ))

loss, accuracy = model.evaluate(X_tst, Y_tst, verbose=0)
print 'Loss:'+str(loss)+'\nAccuracy:'+str(accuracy)
#
#
#
#
#
