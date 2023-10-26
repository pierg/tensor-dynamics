# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 20:57:14 202

this is siilar to v10.py, except I normalize the data in the stanard NN way

Both use the new data set in Dataset2/F1


@author: alexa
"""

import numpy as np
import pickle
from pathlib import Path
from tensorflow.keras import layers, models, Model
from tensorflow.keras.models import clone_model
from tensorflow.keras.layers import (
    Dense,
    Input,
    LSTM,
    concatenate,
    Flatten,
    Dropout,
    Conv2D,
    MaxPooling2D,
    AveragePooling2D,
)
from tensorflow.keras.utils import Sequence
import glob

fn = glob.glob("/global/scratch/users/edkinigstein/Dataset2/F1/MM_Data_set_1_10_18_23*")

nn_data = []
predictions = []

num_file = 15

for i in range(num_file):
    data = pickle.load(open(fn[i], "rb"))
    for j in range(len(data)):
        nn_data.append(data[j]["NN_eValue_Input"] * data[j]["NN_eVector_Input"])
        predictions.append(data[j]["NN_Prediction"].flatten())

nn_data = np.array(nn_data)
predictions = np.array(predictions)

# here I normalize the data in the standard NN way
for i in range(nn_data.shape[1]):
    for j in range(nn_data.shape[2]):
        nn_data[:, i, j] = (nn_data[:, i, j] - np.mean(nn_data[:, i, j])) / np.std(
            nn_data[:, i, j]
        )


nn_data = nn_data.reshape(
    nn_data.shape[0], nn_data.shape[1], nn_data.shape[2], 1
)  # there's a neater way of doing this!

nn_length = nn_data.shape[1]


dropout_level = 0.2

cf = [1, 1, 1]  # convolutional factors

model_input = Input(shape=(nn_length, 3, 1))

model = Conv2D(64, (3, cf[1]), activation="relu")(model_input)
# model = MaxPooling2D((2, 1))(model)
##model = Dropout(dropout_level)(model)
# model = Conv2D(128, (3, cf[2]), activation='relu')(model)
# model = MaxPooling2D((2, 1))(model)
##model = Dropout(dropout_level)(model)
model = Flatten()(model)
model = Dense(128, activation="relu")(model)
# model = Dropout(dropout_level)(model)
model = Dense(128, activation="relu")(model)
model = Dense(64, activation="relu")(model)
model = Dense(32, activation="relu")(model)
# model = Dropout(dropout_level)(model)
model = Dense(12, activation="linear")(model)

model = Model(inputs=model_input, outputs=model)
model.compile(
    optimizer="adam", loss="mean_squared_error", metrics=["MeanSquaredError"]
)  # accuracy is not the right metric here

model.summary()

model.fit(
    nn_data, predictions, epochs=600, verbose=2, validation_split=0.05
)  # validation_split automatically generates validation data; all data is shuffled by default

test_ev_data = []
test_v_data = []
test_predictions = []

data = pickle.load(open(fn[199], "rb"))

for j in range(4000):
    test_ev_data.append(data[j]["NN_eValue_Input"])
    test_v_data.append(data[j]["NN_eVector_Input"])
    test_predictions.append(data[j]["NN_Prediction"].flatten())

model.evaluate([test_v_data[:10], test_ev_data[:10]], test_predictions[:10])
