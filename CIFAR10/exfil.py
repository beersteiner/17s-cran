import numpy as np
import keras


from keras.datasets import cifar10
(xtest, yjunk),(xjunk2, yjunk2) = cifar10.load_data()
xtest = xtest[0]


IMG_SHAPE = (32,32,3)
N_MAL_IMG = 8192
FILEPATH = 'model_bad_2e.hdf5'

np.random.seed(47405)
Xmal = np.random.choice(a=255, size=np.insert(IMG_SHAPE, 0, N_MAL_IMG), replace=True)

model = keras.models.load_model(FILEPATH)

Y = model.predict(x=Xmal) # gives prediction vectors
Y_cat = Y.argmax(axis=1)  # gives predicted most likely category
