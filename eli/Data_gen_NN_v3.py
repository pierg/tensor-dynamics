# -*- coding: utf-8 -*-
"""
Created on Sun October 30th 2023

@author: Eli K + Fengchen
"""

import numpy as np
import pickle
from pathlib import Path
from tensorflow.keras import layers, models, Model
from tensorflow.keras.models import clone_model
from tensorflow.keras.layers import Dense, Input, LSTM, concatenate, Flatten, Dropout, Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.utils import Sequence
import glob
import tensorflow as tf
import time

fn=glob.glob('/global/scratch/users/edkinigstein/Dataset2/F1/MM_Data_set_1_10_18_23_*')

class CustomDataGenerator(Sequence):
    def __init__(self, file_list):
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        data = pickle.load(open(self.file_list[idx],'rb'))
        
        nn_data = []
        predictions = []

        for j in range(len(data)):
            nn_data.append(data[j]['NN_eValue_Input']*data[j]['NN_eVector_Input'])
            predictions.append(data[j]['NN_Prediction'].flatten())

        predictions = np.array(predictions)
        nn_data = np.array(nn_data)
        
        #normalize the data across datasets
        for i in range(nn_data.shape[1]):
            for j in range(nn_data.shape[2]):
                nn_data[:,i,j]=(nn_data[:,i,j]-np.mean(nn_data[:,i,j]))/np.std(nn_data[:,i,j])

        #now split the dataset in each file into NB chunks
        NB= 4
        BS=int(len(data)/NB)
        sod=nn_data[:BS,:,:].shape
        nn_d_i=np.zeros((sod[0],sod[1], sod[2]))
        NND= []
        NNP= []
        
        for i in range(NB):
            nn_d_i=nn_data[i*BS:(i+1)*BS,:,:]
            NND.append(nn_d_i.reshape(sod[0],sod[1],sod[2],1))
            
            NNP.append(predictions[i*BS:(i+1)*BS,:])
            
        
#        nn_data = np.array(nn_data).reshape(nn_data.shape[0], nn_data.shape[1], nn_data.shape[2], 1)
        return NND, NNP

data_gen = CustomDataGenerator(fn)

#rememebr loading each file results in the creating of an object called nn_data, and predicitons
#when you call nn_length = data_gen["File Number"] it should return a list [nn_data, predictions]
# when you call nn_length = data_gen["File Number"][0/1] it will return nn_data/predictions
#nn_data is an array of shape [num dta pnts in file][eigenvector length][num eignevectors][1]


nn_length = data_gen[0][0][0].shape[1] # Assuming all files have the same shape.
Num_Vectors = data_gen[0][0][0].shape[2] # this is the number of eignevectors
strategy = tf.distribute.MirroredStrategy() # Utilizing all GPUs or CPUs,

with strategy.scope():
    model_input = Input(shape=(nn_length,Num_Vectors,1))
    model = Conv2D(32, (3, 1), activation='relu')(model_input)
    model = Flatten()(model)
    model = Dense(128, activation='relu')(model)
    model = Dense(128, activation='relu')(model)
    model = Dense(64, activation='relu')(model)
    model = Dense(12, activation='linear')(model)
    model = Model(inputs=model_input, outputs=model)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['MeanSquaredError'])

model.summary()

for epoch in range(100): # number of epochs
    print(f"Epoch {epoch+1}/100")
    loss_avg, val_loss_avg = 0, 0
    t_st=time.time()
    for i in range(len(data_gen)): # here we loop over all of the files in the list for each epoch
#        print(f"Loading dataset {i}")
        
        nn_data1, predictions1,= data_gen[i]

        for q in range(len(nn_data1)):
            split_index = int(0.95*nn_data1[q].shape[0])  # using 95% of data for training and 5% for validation
            train_data1, val_data1 =       nn_data1[q][:split_index,:,:,:], nn_data1[q][split_index:,:,:,:]
            train_preds1, val_preds1 = predictions1[q][:split_index,:], predictions1[q][split_index:,:]
            loss1, _ = model.train_on_batch(train_data1, train_preds1)
            loss_avg += loss1/len(nn_data1)
            val_loss1, _ = model.test_on_batch(val_data1, val_preds1)
            val_loss_avg += val_loss1/len(nn_data1)
       
        
        
        
    total_time=time.time()- t_st
    loss_avg /= len(data_gen)
    val_loss_avg /= len(data_gen)
    print(f"Training - Loss: {loss_avg:.4f}")
    print(f"Validation - Loss: {val_loss_avg:.4f}")
    print(f"This epoch took: {total_time:.2f} seconds")
