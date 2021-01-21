
import os
import numpy as np

from keras.models import Model
from keras.layers import GRU, Input, Dense
from keras import losses
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.models import load_model
#from keras.callbacks import EarlyStopping, ModelCheckpoint

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

from sklearn.utils import class_weight
from sklearn.metrics import classification_report

KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'CPU':1,'GPU':1})))

datapath = ''
modelpath = ''

#############################  HYPERPARAMETERS ##########################################
ntr = 3; nte = 3; nva = 5   #duration of one utterance 'in seconds' for - training, validation and testing files 
EPOCHS = 40
DROP = 0.2
REC_DROP = 0.2
BATCH_SIZE = 64
#VAL_SPLIT = 0.2
OPTIMIZER = Adam(1e-3)
l2 = l2(1e-2)

print("Loading npz files...")

x_train = np.load(datapath+'mfcc_train.npz')
x_test = np.load(datapath+'mfcc_test.npz')
x_val = np.load(datapath+'mfcc_val.npz')
y_train = np.load(datapath+'l_train.npz')
y_test = np.load(datapath+'l_test.npz')
y_val = np.load(datapath+'l_val.npz')


print ("\nData loaded. Converting to arrays...")
xtr = x_train['arr_0']
xte = x_test['arr_0']
xva = x_val['arr_0']
ytr = y_train['arr_0']
yte = y_test['arr_0']
yva = y_val['arr_0']
print ("\nNpz object converted to arrays")
xtr = np.transpose(xtr)
xte = np.transpose(xte)
xva = np.transpose(xva)

#Demarcating language-wise dataset and then creating batches of 3ms
print ("\nShapes of loaded datasets:",xtr.shape, xte.shape,xva.shape,ytr.shape,yte.shape,yva.shape)
unique_train , counts_train = np.unique(ytr, return_counts= True)
unique_test, counts_test = np.unique(yte, return_counts= True)
unique_val, counts_val = np.unique(yva, return_counts= True)
print ("\nPrinting unique(label)values:",unique_train , counts_train, unique_test, counts_test, unique_val, counts_val)

xtr_eng = xtr[:counts_train[0]] 
xtr_hin = xtr[counts_train[0]:counts_train[0]+counts_train[1]]
xtr_mand = xtr[counts_train[0]+counts_train[1]:counts_train[0]+counts_train[1]+counts_train[2]] 
print ("\nShapes of training dataset language-wise: ","e:",xtr_eng.shape,"h:",xtr_hin.shape,"m:",xtr_mand.shape)

xte_eng = xte[:counts_test[0]] 
xte_hin = xte[counts_test[0]:counts_test[0]+counts_test[1]]
xte_mand = xte[counts_test[0]+counts_test[1]:counts_test[0]+counts_test[1]+counts_test[2]] 
print ("Shape of test dataset language-wise: ",xte_eng.shape,xte_hin.shape,xte_mand.shape)

xva_eng = xva[:counts_val[0]] 
xva_hin = xva[counts_val[0]:counts_val[0]+counts_val[1]]
xva_mand = xva[counts_val[0]+counts_val[1]:counts_val[0]+counts_val[1]+counts_val[2]] 
print ("Shape of validation dataset language-wise: ",xva_eng.shape,xva_hin.shape,xva_mand.shape)
   
############################# ------------------------------ ------------------------------ ------------------------------
#############################  Grouping mfcc vectors into sequences of ntr,nte,nva-seconds #################################
#------------------------------ ------------------------------ ------------------------------ #############################
print('\n Grouping data into n-second sequences...')
le = int(np.floor(len(xtr_eng)/(ntr*100))); lh = int(np.floor(len(xtr_hin)/(ntr*100))); lm = int(np.floor(len(xtr_mand)/(ntr*100)))
# ntr*100, nva*100, nte*100 since, for a 3 sec sample with 10 ms mfcc hop-length we will get 300 samples (298.5 to be exact for 25 ms long mfcc samples. (3000 - (25-10) )/10 So, with 300 samples we are going a little above 3 milli seconds).
print("Length of training set after", str(ntr), "s truncations: e,h,m:",le,lh,lm)
#le should be 10904
xtr_e = np.reshape(xtr_eng[:le*ntr*100],( le ,ntr*100 ,64))
xtr_h = np.reshape(xtr_hin[:lh*ntr*100],( lh ,ntr*100 ,64))
xtr_m = np.reshape(xtr_mand[:lm*ntr*100],( lm ,ntr*100 ,64))

ytr_e = np.repeat(0,le)
ytr_h = np.repeat(1,lh)
ytr_m = np.repeat(2,lm)

le = int(np.floor(len(xte_eng)/(nte*100))); lh = int(np.floor(len(xte_hin)/(nte*100))); lm = int(np.floor(len(xte_mand)/(nte*100)))
print("Length of test set after", str(nte), "s truncations: ",le,lh,lm)
#le should be 10904
xte_e = np.reshape(xte_eng[:le*nte*100],( le ,nte*100 ,64))
xte_h = np.reshape(xte_hin[:lh*nte*100],( lh ,nte*100 ,64))
xte_m = np.reshape(xte_mand[:lm*nte*100],( lm ,nte*100 ,64))

yte_e = np.repeat(0,le)
yte_h = np.repeat(1,lh)
yte_m = np.repeat(2,lm)

le = int(np.floor(len(xva_eng)/(nva*100))); lh = int(np.floor(len(xva_hin)/(nva*100))); lm = int(np.floor(len(xva_mand)/(nva*100)))
print("Length of test set after", str(nva), "s truncations: ",le,lh,lm)
#le should be 10904
xva_e = np.reshape(xva_eng[:le*nva*100],( le ,nva*100 ,64))
xva_h = np.reshape(xva_hin[:lh*nva*100],( lh ,nva*100 ,64))
xva_m = np.reshape(xva_mand[:lm*nva*100],( lm ,nva*100 ,64))

yva_e = np.repeat(0,le)
yva_h = np.repeat(1,lh)
yva_m = np.repeat(2,lm)

#Concatenating reshaped Data:
mfcc_train = np.concatenate((xtr_e,xtr_h,xtr_m))
label_train = np.concatenate((ytr_e,ytr_h,ytr_m))  
mfcc_test = np.concatenate((xte_e,xte_h,xte_m))
label_test = np.concatenate((yte_e,yte_h,yte_m))
mfcc_val = np.concatenate((xva_e,xva_h,xva_m))
label_val = np.concatenate((yva_e,yva_h,yva_m))
cw = class_weight.compute_class_weight('balanced',np.unique(label_train),label_train)
print ("Shape of the final dataset: mfcc_train",mfcc_train.shape,"label_train: ",label_train.shape, "mfcc_test:",mfcc_test.shape, "label_test:",label_test.shape,"mfcc_val:",mfcc_val.shape, "label_val:",label_val.shape  )


############################################################################################
#############################  Define/Build Model ##########################################
############################################################################################
print ('\nBuilding Model...')
main_input = Input (shape=(None, 64), name = 'main_input')
dnn1 = Dense(32, kernel_regularizer = l2, name = 'dnn1')(main_input)
gru1 =  GRU(64, dropout = DROP, recurrent_dropout = REC_DROP ,return_sequences = True,kernel_regularizer=l2, recurrent_regularizer=l2, name = 'gru1' )(dnn1)          
gru2 = GRU(32, dropout = 0.0, recurrent_dropout = 0.0 ,return_sequences = False, kernel_regularizer=l2, recurrent_regularizer=l2, name = 'gru2' )(gru1)
rnn_output = Dense (3, activation = 'softmax', name = 'rnn_output')(gru2)

model = Model(inputs = main_input, outputs = rnn_output)

print ('\nCompiling Model...')
model.compile (loss = 'sparse_categorical_crossentropy', optimizer = OPTIMIZER, metrics = ['sparse_categorical_accuracy'])
print(model.summary())

############################################################################################
#############################  Train and save model #######################################
############################################################################################
print('\nTraining...')    
history = model.fit(mfcc_train, label_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data = (mfcc_val,label_val), shuffle=1,class_weight = cw)
model.save(modelpath+'hw5p1.h5')


############################################################################################
#############################  Test model: Test Accuracy and Class-f1 report #######################################
############################################################################################
print('\nTesting...')
model = load_model(modelpath+'hw5p1.h5')
score = model.evaluate(mfcc_test, label_test)
print('Test accuracy of Model for '+str(nte)+'sec samples = {0}'.format(100*score[1]))
y_prob = model.predict(mfcc_test)
y_pred = y_prob.argmax(axis=-1)
print(classification_report(label_test,y_pred))
