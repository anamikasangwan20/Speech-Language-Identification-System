#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 19:24:05 2019

@author: anamika
"""
import h5py
import numpy as np
import os

from keras.models import Model
from keras.layers import GRU
from keras.layers import Input
from keras.layers import Dense
from keras.optimizers import Adam
from keras.regularizers import l2
# from keras import metrics
from keras import losses
from keras.models import load_model

from sklearn.metrics import classification_report

datapath = ''
modelpath = ''

############################################################################################
######### Generating MFCC from WAV TEST FILES: COMMENT THIS IF USING mfcc.npz, label.npz files ################
################## UPDATE [LINE 36] ##########################################################################
#import json
#import librosa
#
#data_file = open (datapath+'train_files.json') 
#data = json.load(data_file)
#
#d = list(data)
#lables = list(data.values())
#filename = list(data.keys())
#
#mat_test = np.empty((64,0))
#label_test = np.empty((1,0),int)
#
#for item in filename:
#    y, sr = librosa.load(datapath+item, sr=16000)
#    mat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=64, n_fft=int(sr*0.025), hop_length=int(sr*0.010))
#    mat_test = np.append(mat_test,mat,1)
#    label = np.repeat(data[item],mat.shape[1])
#    label_test = np.append(label_test,label)
#    
#print('Saving Test data...')
#np.savez_compressed (datapath+'mfcc_test',mat_test)
#print("Test Data saved.")
#np.savez_compressed (datapath+'l_test',label_test) 
#print("Test labels saved")

############################################################################################
############################################################################################
############################################################################################
print("Loading Test npz files...\n")
x_test = np.load('mfcc_test.npz')
y_test = np.load('l_test.npz')

#LOAD MODEL:
print('Loading trained Model...\n')
model = load_model('hw5p1.h5')

print ("Data loaded. Converting to arrays...\n")
xte = x_test['arr_0']
yte = y_test['arr_0']
print ("Npz object converted to arrays.\n")
xte = np.transpose(xte)

############################################################################################
############## Test model: Test Accuracy and Class-f1 report on 25ms samples ################
############################################################################################
#nte = 3 #length of test samples in 'seconds'
#
#unique_test, counts_test = np.unique(yte, return_counts= True)
#xte_eng = xte[:counts_test[0]] 
#xte_hin = xte[counts_test[0]:counts_test[0]+counts_test[1]]
#xte_mand = xte[counts_test[0]+counts_test[1]:counts_test[0]+counts_test[1]+counts_test[2]] 
#print ("Shape of test dataset language-wise: ",xte_eng.shape,xte_hin.shape,xte_mand.shape)
#
#print('\n Grouping data into n-second sequences...')
#le = int(np.floor(len(xte_eng)/(nte*100))); lh = int(np.floor(len(xte_hin)/(nte*100))); lm = int(np.floor(len(xte_mand)/(nte*100)))
#print("Length of test set after", str(nte), "s truncations: ",le,lh,lm)
##le should be 10904
#xte_e = np.reshape(xte_eng[:le*nte*100],( le ,nte*100 ,64))
#xte_h = np.reshape(xte_hin[:lh*nte*100],( lh ,nte*100 ,64))
#xte_m = np.reshape(xte_mand[:lm*nte*100],( lm ,nte*100 ,64))
#
#yte_e = np.repeat(0,le)
#yte_h = np.repeat(1,lh)
#yte_m = np.repeat(2,lm)
#
#mfcc_test = np.concatenate((xte_e,xte_h,xte_m))
#label_test = np.concatenate((yte_e,yte_h,yte_m))
#
#print('\nTesting...')
#model = load_model(modelpath+'hw5p1.h5')
#score = model.evaluate(mfcc_test, label_test)
#print('Test accuracy of Model for '+str(nte)+'sec samples = {0}'.format(100*score[1]))
#y_prob = model.predict(mfcc_test)
#y_pred = y_prob.argmax(axis=-1)
#print(classification_report(label_test,y_pred))

############################################################################################
#############################  Test Streaming model: #Correctly Classified ##################
############################################################################################

#Building Sequential Model:
DROP = 0.2
REC_DROP = 0.2
OPTIMIZER = Adam(1e-3)

print('Building Model...\n')
streaming_input = Input(name='streaming_input', batch_shape=(1,1,64))
dnn1 = Dense(32, name = 'dnn1')(streaming_input)
gru1 =  GRU(64, dropout = DROP, recurrent_dropout = REC_DROP ,return_sequences = True, stateful = True, name = 'gru1' )(dnn1)          
gru2 = GRU(32, dropout = 0.0, recurrent_dropout = 0.0 ,return_sequences = False, stateful = True, name = 'gru2' )(gru1)
rnn_output = Dense (3, activation = 'softmax', name = 'rnn_output')(gru2)


print('Compiling Model...\n')
streaming_model = Model(inputs=streaming_input, outputs=rnn_output)
streaming_model.compile (loss = 'sparse_categorical_crossentropy', optimizer = OPTIMIZER, metrics = ['sparse_categorical_accuracy'])

print('Loading Model Weights...\n')
old_weights = model.get_weights()
streaming_model.set_weights(old_weights)

print('Prediction starts....\n')
correct = 0
samples = 0
for i in range (len(xte)):
    one_sequence = xte[i]
    stream_predictions = streaming_model.predict(np.reshape(one_sequence, (1,1,64)), batch_size = 1 )
    label_prediction = np.argmax(stream_predictions)
    samples+=1
    if label_prediction == yte[i]:
        correct+=1
    if ((i%1000 == 0) & (i!=0)):
        streaming_model.reset_states()
    if ((i%10000 == 0) & (i!=0)):
        print(stream_predictions,label_prediction,yte[i],round(correct*100/samples,1))
        break
#print('Number of Correct Predictions out of ' +str(len(xte))+' are: '+str(correct))
#stream_acc = correct/float(len(xte))
print('Number of Correct Predictions out of ' +str(samples)+' are: '+str(correct))
stream_acc = correct/float(samples)
print ('Thus, Test Accuracy is:'+str(stream_acc))



