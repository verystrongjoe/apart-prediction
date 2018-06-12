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
"""
preparing data
"""
X_train, X_test, y_train, y_test = util.load_data_set()

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
model.compile(optimizer='rmsprop')

model.fit([X_train, X_train], [y_train])




