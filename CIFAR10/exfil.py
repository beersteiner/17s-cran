import numpy as np
import keras

IMG_SHAPE = (32,32,3)
N_MAL_IMG = 8192
FILEPATH = 'model_bad_2e.hdf5'

np.random.seed(47405)
Xmal = np.random.choice(a=255, size=np.insert(IMG_SHAPE, 0, N_MAL_IMG), replace=True)

model = keras.models.load_model(FILEPATH)
Ymal = model.predict(x=Xmal)
