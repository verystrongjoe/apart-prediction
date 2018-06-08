import pandas as pd

import  keras
from keras.models import Sequential
from keras.layers import Conv1D, Flatten

N_FEATURES = 20
N_TIME_WINDOW = 6
INPUT_SHAPE = (N_FEATURES, N_TIME_WINDOW)

df_61 = pd.read_csv('data\\kab61.csv')
df_71 = pd.read_csv('data\\kab71.csv')
df_61_filtered = df_61[['AreaName', 'YYYYMMDD', 'InfoType2', 'Values']]
df_71_filtered = df_71[['AreaName', 'YYYYMMDD', 'InfoType2', 'Values']]
df_x = df_61_filtered.append(df_71_filtered)
df_y = pd.read_csv('data\\y.txt', delimiter='\t', encoding='MS949')

model = Sequential()
model.add(Conv1D(32, kernel_size=(6), activation='relu', input_shape=INPUT_SHAPE))
model.add(Flatten())

