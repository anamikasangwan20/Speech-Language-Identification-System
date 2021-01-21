import json
#import os
import librosa
import numpy as np

datapath = '/Users/anamika/Documents/Python/DL_hw5/'

data_file = open (datapath+'train_files.json') 
data = json.load(data_file)

d = list(data)
lables = list(data.values())
filename = list(data.keys())

#Creating new val and train files with no overlapping speakers:

#Training-Validation partition with non-overlapping speakers:

#13 test speakers-Fisrt 7 are mandarin-english and last 6 are hindi-english
test_speakers = ['-80-','-79-','-78-','-54-','-41-','-17-','-09-','-77-','-72-','-66-','-51-','-36-','-03-']
#12 val speakers - First 6 are mandarin-english and last 6 are hindi-english
val_speakers = ['-04-','-16-','-21-','-29-','-34-','-62-','-23-','-38-','-42-','-61-','-07','-12-']

print("Checking for speaker overlap:",any(c in test_speakers for c in val_speakers),"-->Should be False")

mat_train = np.empty((64,0))
label_train = np.empty((1,0),int)
mat_test = np.empty((64,0))
label_test = np.empty((1,0),int)
mat_val = np.empty((64,0))
label_val = np.empty((1,0),int)
i=1
for item in filename:
    if any(c in item for c in test_speakers):
        y, sr = librosa.load(datapath+'train/'+item, sr=16000)
        mat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=64, n_fft=int(sr*0.025), hop_length=int(sr*0.010))
        mat_test = np.append(mat_test,mat,1)
        label = np.repeat(data[item],mat.shape[1])
        label_test = np.append(label_test,label)
        print (i,'$$TEST',item,'language =',label[0], mat.shape, label_test.shape)
    elif any(c in item for c in val_speakers):
        y, sr = librosa.load(datapath+'train/'+item, sr=16000)
        mat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=64, n_fft=int(sr*0.025), hop_length=int(sr*0.010))
        mat_val = np.append(mat_val,mat,1)
        label = np.repeat(data[item],mat.shape[1])
        label_val = np.append(label_val,label)
        print (i,'**VAL',item,'language =',label[0], mat.shape, label_val.shape)
    else:
        y, sr = librosa.load(datapath+'train/'+item, sr=16000)
        mat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=64, n_fft=int(sr*0.025), hop_length=int(sr*0.010))
        mat_train = np.append(mat_train,mat,1)
        label = np.repeat(data[item],mat.shape[1])
        label_train = np.append(label_train,label)
        print (i,'TRAIN',item,'language =',label[0], mat.shape, label_train.shape)

print("Saving Training data...")
np.savez_compressed (datapath+'mfcc1_train',mat_train)
print("Trainging Data Saved. Saving Test data...")
np.savez_compressed (datapath+'mfcc_test',mat_test)
print("Test Data saved.")
np.savez_compressed (datapath+'mfcc_val',mat_val)
print("Validation Data saved.")
np.savez_compressed (datapath+'l_train',label_train)
print("Train labels saved")
np.savez_compressed (datapath+'l_test',label_test) 
print("Test labels saved")
np.savez_compressed (datapath+'l_val',label_val) 
print("Validation labels saved")