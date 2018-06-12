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

"""
preparing data
create training data 70%, testing data 30%
"""
X_train, X_test, y_train, y_test = util.load_data_set(30, 33)
m = preprocessing.get_sido_onehot_map()

"""
merge two other neural networks
https://statcompute.wordpress.com/2017/01/08/an-example-of-merge-layer-in-keras/
https://nhanitvn.wordpress.com/2016/09/27/a-keras-layer-for-one-hot-encoding/
"""
d_features = Input(shape=(1, config.N_TIME_WINDOW, config.N_FEATURES), name="features")
d_sido = Input(shape=(15, ), name="sido_onehot")

l1 = Conv1D(32, kernel_size=(12), activation='relu')(d_features)
l2 = Conv1D(64, kernel_size=(6), activation='relu')(l1)
l2_flat = Flatten()(l2)

d_concatenated = Concatenate([l2_flat, d_sido])
output = Dense(3, activation='softmax')(d_concatenated)

model=Model(inputs=[d_features,d_sido],outputs=output)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# training
model.fit( [X_train[0:config.N_FEATURES], m[X_train[config.N_FEATURES][0]]], [y_train])

# save weight
model.save_weights('{}.hdf5'.format('uk'))

# finished..