import numpy as np
import keras
from keras.utils import np_utils
import png
import argparse


# Set up argument parser
parser = argparse.ArgumentParser(description='Exfiltrate secret training info!')
parser.add_argument('-f', '--file', metavar='filepath', type=str, default='',
        help='Model file to use', required=True)
parser.add_argument('-t', '--test', action='store_true', default=False,
        help='Generate original data and evaluate model performance', required=False)
args = parser.parse_args()




# GLOBALS

# A priori globals
N_XFIL_IMG = 1          # number of trng images to exfil
FILEPATH = args.file    # path to model file
TEST = args.test        # if true, enables a test mode of this script

# Calculated globals
N_CAT = None            # number of categories
IMG_SHAPE = None        # shape of input images
N_MAL_IMG = None        # number of malicious imgs needed to produce ecoded labels
N_BITS_LAB = None       # number of bits we can get from a single label



# FUNCTIONS

# pre-process image data
def preproc_x(d):
    d = d.astype('float32')     # convert to float
    d -= 128.0                  # center around 128
    d /= 128.0                  # normalize -1 to 1
    return d

# Post-process to restore RGB image
def postproc_x(d):
    d = d.astype('float32')     # convert to float
    d *= 128.0                  # scale back to -128-128
    d += 128.0                  # shift to 0-255
    return d.astype('uint8')    # conv to uint8

# pre-process label data
def preproc_y(d):
    return np_utils.to_categorical(d, N_CAT)    # conv dec categories to binary vectors

# This is just a wrapper to write PNG files
def writePNG(filename, a):
    f = open(filename, 'wb')
    w = png.Writer(size=IMG_SHAPE[:2])
    w.write(f, a)
    f.close()
    return 0


# Encode an image array into categorical binary array
def imgToLabs(img):
    n_bits = img.size * 8  # number of bits to encode
    res = np.unpackbits(img)[:n_bits]  # bit representation of img
    rem = res.shape[0] % N_BITS_LAB  # number of remaining bits if not round
    if rem: res = np.concatenate((res, np.zeros(r)))  # zero pad to make round
    res = res.reshape((res.size / N_BITS_LAB, N_BITS_LAB))  # num of false images x num bits per label
    res = np.right_shift(np.packbits(res, axis=-1), 8-N_BITS_LAB)  # convert to dec category numbers
    res = np_utils.to_categorical(res, N_CAT)  # convert to categorical binary vectors
    return res

# Decode labels back into image format
def labsToImg(labs):
    n_xfil_bits = N_XFIL_IMG * np.prod(IMG_SHAPE) * 8 # should be 24576
    res = labs.argmax(axis=1).astype('uint8')  # conv probs to decimal cat nums
    # num of labels x num of bits per label
    res = np.unpackbits(res, axis=-1).reshape((N_MAL_IMG, 8))[:,8-N_BITS_LAB:]
    res = res.reshape((1, res.size))  # flatten
    res = res[:n_xfil_bits]  # extract only the number of bits needed
    res = np.packbits(res).reshape(np.insert(IMG_SHAPE, 0, N_XFIL_IMG))  # repack and reshape to img format
    return res



# MAIN

# load the model and update globals
model = keras.models.load_model(FILEPATH)
N_CAT = model.layers[-1].output_shape[1]    # get num of categories from output shape
IMG_SHAPE = model.layers[0].input_shape[1:] # get img shape from input layer
# number of malicious images calculated from img shape, num categories, and num of imgs to exfil
N_MAL_IMG = int(np.ceil(N_XFIL_IMG * np.prod(IMG_SHAPE) * 8) / np.floor(np.log2(N_CAT)))
N_BITS_LAB = int(np.floor(np.log2(N_CAT)))

# Re-create the malicious training images
np.random.seed(47405)   # this seed must be consistent with that of training model
# Generate a random array of 0-255 values and shape them into images
xmal = np.random.choice(a=255, size=np.insert(IMG_SHAPE, 0, N_MAL_IMG), replace=True)
xmal = preproc_x(xmal)  # perform deterministic pre-processing (also consistent with trainer)

# This is for test only, to make sure encode/decode is working
if TEST:
    from keras.datasets import cifar10
    (xtrn, ytrn),(xtst, ytst) = cifar10.load_data()
    xtestimg = xtrn[0]
    writePNG('original.png', xtestimg.reshape((IMG_SHAPE[0], np.prod(IMG_SHAPE[1:]))).astype('uint8'))
    Y = imgToLabs(xtestimg)  # Cheat predictions for test purposes
    print 'Evaluating model against training data'
    loss, acc = model.evaluate(x=preproc_x(xtrn), y=preproc_y(ytrn), verbose=1)
    print 'Loss:' + str(loss) + '\nAccuracy:' + str(acc)
    print 'Evaluating model against test data'
    loss, acc = model.evaluate(x=preproc_x(xtst), y=preproc_y(ytst), verbose=1)
    print 'Loss:' + str(loss) + '\nAccuracy:' + str(acc)
    print 'Evaluating model against malicious data'
    loss, acc = model.evaluate(x=xmal, y=Y, verbose=1)
    print 'Loss:' + str(loss) + '\nAccuracy:' + str(acc)
else:
    # Use model to get predictions
    Y = model.predict(x=xmal) # gives prediction vectors

    
# Decode the predictions to obtain the secret data
Xsec = labsToImg(Y)

# Write exfiltrated data to a png file
writePNG(FILEPATH+'_exfil.png', Xsec.reshape((IMG_SHAPE[0],np.prod(IMG_SHAPE[1:]))).astype('uint8'))





