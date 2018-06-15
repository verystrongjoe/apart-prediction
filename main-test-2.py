import  keras
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv1D, Flatten, Dense
from keras.layers import Input, Lambda, Concatenate
from keras import backend as K
from keras.models import Model
import numpy as np
from sklearn.model_selection import train_test_split
import util
import config
import preprocessing
import pickle
import os

"""
preparing data
create training data 70%, testing data 30%
"""
config.N_FEATURES = len(preprocessing.get_fetures_nm_list())
PICKLE_FILE_NAME = 'TEST2'
MODEL_FILE_NAME = 'TEST2'
mode = 2
if os.path.isfile('{}.pickle'.format(PICKLE_FILE_NAME)) :
    f = open('{}.pickle'.format(PICKLE_FILE_NAME), 'rb')
    print('loading pickle file from disk')
    l = pickle.load(f)
    X_train, X_test, y_train, y_test, m = l[0], l[1], l[2], l[3], l[4]
else:
    X_train, X_test, y_train, y_test = util.load_data_set(30, 33, mode=2)
    m = preprocessing.get_sido_onehot_map()
    f = open('{}.pickle'.format(PICKLE_FILE_NAME), 'wb')
    pickle.dump([X_train, X_test, y_train, y_test, m], f)
    print('finished to dump pickle file from disk')

"""
merge two other neural networks
https://statcompute.wordpress.com/2017/01/08/an-example-of-merge-layer-in-keras/
https://nhanitvn.wordpress.com/2016/09/27/a-keras-layer-for-one-hot-encoding/
"""
# d_features = Input(shape=(1, config.N_FEATURES, config.N_TIME_WINDOW), name="features")
# d_sido = Input(shape=(len(preprocessing.get_sido_nm_list()), ), name="sido_onehot")
d_features = Input(shape=(config.N_TIME_WINDOW, config.N_FEATURES), name="features")
# d_sido = Input(shape=(len(preprocessing.get_sido_nm_list()),), name="sido_onehot")

d1 = Dense(32)(d_features)
d2 = Dense(32)(d1)
d3 = Dense(32)(d2)
output = Dense(3, activation='softmax')(d3)

model = Model(inputs=d_features,outputs=output)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# training
model.fit(
    [np.array(X_train[:,:,0:config.N_FEATURES]),  np.array([m[preprocessing.get_sido_nm_list()[int(r)]] for r in X_train[:,0,config.N_FEATURES]]) ],
    np.array(keras.utils.to_categorical(y_train, 3)), epochs=20, batch_size=10)

# save weight
model.save_weights('{}.hdf5'.format(MODEL_FILE_NAME))

# finished and predict using training model above!!
score = model.evaluate(
    [np.array(X_test[:,:,0:config.N_FEATURES]),  np.array([m[preprocessing.get_sido_nm_list()[int(r)]] for r in X_test[:,0,config.N_FEATURES]]) ],
    np.array(keras.utils.to_categorical(y_test, 3)), verbose=0)


print('Test loss:', score[0])
print('Test accuracy:', score[1])
