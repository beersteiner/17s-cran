#!/usr/bin/env python

# Logistic regression model via keras found at:
#   https://medium.com/@the1ju/simple-logistic-regression-using-keras-249e0cc9a970


import argparse
# Parse arguments for hyperparameters
parser = argparse.ArgumentParser(
        description='Generate model to learn Good/Bad model responses')
parser.add_argument('-f', '--files', type=argparse.FileType('r'), nargs='+')
parser.add_argument('-a','--analyze', action='store_true', default=False,
                            help='Perform data analysis.', required=False)
#parser.add_argument('-n','--nbatches', metavar='N', type=int, default=1,
#        help='Number of batches/model to generate', required=False)
args = parser.parse_args()



import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import ModelCheckpoint, CSVLogger
import numpy as np
from atk_features import structArray, ewFeat, analyze


N_CAT = 2
B_SIZE = 16
N_EPOCH = 20


for f in args.files:
    print 'Loading Data...'
    # Get labels, predictions from data file
    Y,X = np.split(np.genfromtxt(f, delimiter=','), [1], axis=1)
    # Represent X as a structured array with field names
    X = structArray(X)
    # Augment structured array with element-wise features
    X = ewFeat(X)
    # Flatten Labels
    Y = np.reshape(Y, (Y.shape[0],))
    if args.analyze: analyze(os.path.split(f.name)[1], X, Y)
    
    
    # Train Attack Model
    # split data into training & test
    split = 0.9 # percent of total data to use as training
    Xtrn = X[:int(X.size*split)]
    Xtst = X[int(X.size*split):]
    Ytrn = keras.utils.to_categorical(Y[:int(Y.size*split)])
    Ytst = keras.utils.to_categorical(Y[int(Y.size*split):])
    
    # Select features
    #feat = Xtrn.dtype.names
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
    
    # Build, compile, fit, evaluate model
    print 'Building Model...'
    model = Sequential()
    model.add(Dense(N_CAT, input_dim=len(feat), activation='softmax'))
    checkpoint = ModelCheckpoint('./atkmodels/atkmodel.hdf5', monitor='val_acc', verbose=True,
            save_best_only=True, save_weights_only=False, mode='auto', period=1)
    csv_logger = CSVLogger('./atkmodels/atkmodel_trng.log')
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    print 'Fitting Model...'
    history = model.fit(
            Xtrn[feat].view(np.float64).reshape(Xtrn.shape + (len(feat),)),
            Ytrn, batch_size=B_SIZE, epochs=N_EPOCH, 
            callbacks=[checkpoint, csv_logger],
            verbose=1, validation_data=(Xtst[feat].view(np.float64).reshape(Xtst.shape + (len(feat),)), Ytst))
    score = model.evaluate(Xtst, Ytst, verbose=0)
    print 'Test score: ' + score[0]
    print 'Test accuracy: ' + score[1]


    print f.name + ' Complete.'
    f.close()
    

print 'Complete!'
