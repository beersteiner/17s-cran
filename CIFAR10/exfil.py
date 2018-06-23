import numpy as np
import keras
from keras.utils import np_utils
import png


# Some global vars
IMG_SHAPE = (32,32,3)
N_XFIL_IMG = 1
FILEPATH = 'model_bad_2e.hdf5'

N_MAL_IMG = int(np.ceil(N_XFIL_IMG * np.prod(IMG_SHAPE) * 8) / np.floor(np.log2(10)))



# dummy test image to make sure we can encode and decode back to original
from keras.datasets import cifar10
(xtest, yjunk),(xjunk2, yjunk2) = cifar10.load_data()
xtestimg = xtest[0]
# Write exfiltrated data to a png file
file = open('original.png','wb')
w = png.Writer(size = IMG_SHAPE[:2])
w.write(file, xtestimg.reshape((IMG_SHAPE[0],np.prod(IMG_SHAPE[1:]))).astype('uint8'))
file.close()
n_xfil_bits = 1 * xtestimg.size * 8 # should be 24576
xtestlabs = np.unpackbits(xtestimg)[:n_xfil_bits]
n = int(np.floor(np.log2(10))) # num of bits we can encode in a label
r = xtestlabs.shape[0] % n # remainder after you group image into encode chunks
if r: xtestlabs = np.concatenate((x, np.zeros(r))) # pad so image can be divided into encodable chunks
xtestlabs = xtestlabs.reshape((xtestlabs.size / n, n)) # number of mal imgs needed to extract secret data x n
xtestlabs = np.right_shift(np.packbits(xtestlabs, axis=-1), 8-n) # convert bit encodings to cat, then right shift
xtestlabs = np_utils.to_categorical(xtestlabs, 10)



#np.random.seed(47405)
#Xmal = np.random.choice(a=255, size=np.insert(IMG_SHAPE, 0, N_MAL_IMG), replace=True)
#model = keras.models.load_model(FILEPATH)
#Y = model.predict(x=Xmal) # gives prediction vectors

# remove this!!!!!
Y = xtestlabs

Xsec = Y.argmax(axis=1).astype('uint8')  # gives predicted most likely category as dec
Xsec = np.unpackbits(Xsec, axis=-1).reshape((N_MAL_IMG, 8))[:,8-n:] # N_MAL_IMG x 3 array
Xsec = Xsec.reshape((1, Xsec.size)) # flatten into a 1D bit array
Xsec = Xsec[:n_xfil_bits] # remove any extra bits (remainder)
# pack as 0-255 values and reshape into image format
Xsec = np.packbits(Xsec).reshape(np.insert(IMG_SHAPE, 0, N_XFIL_IMG)) 

# Write exfiltrated data to a png file
file = open('exfil.png','wb')
w = png.Writer(size = IMG_SHAPE[:2])
w.write(file, Xsec.reshape((IMG_SHAPE[0],np.prod(IMG_SHAPE[1:]))).astype('uint8'))
file.close()





