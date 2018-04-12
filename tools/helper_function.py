# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 12:35:00 2018

@author: Cyrus DSouza
"""
import numpy as np
import os
from random import randint

ROOT = "D:/Cyrus/Implementations//Sentiment-Analysis-with-LSTM/"


def getTrainBatch(ids,batchSize, maxSeqLength):

    labels = []
    arr = np.zeros([batchSize, maxSeqLength+1])
    for i in range(batchSize):
        if(i%2 == 0):
            num = randint(1,15300)
            labels.append([1,0])
        else:
            num = randint(13233, 15309)
            labels.append([0,1])
        print(arr.shape)
        arr[i] = ids[num-1:num].reshape(0,arr[i].shape[1])
        
    return arr, labels 


def getTestBatch(ids, batchSize, maxSeqLength):

    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        num = randint(11499,13499)
        if (num <= 12499):
            labels.append([1,0])
        else:
            labels.append([0,1])
            
        arr[i] = ids[num-1:num]
        
    return arr, labels    


def find_max_length(positiveFiles, negativeFiles):
    
    numWords = []
    for pf in positiveFiles:
        with open(pf, "r", encoding = 'utf-8')as f:
            line = f.readline()
            counter = len(line.split())
            numWords.append(counter)
    print("Positive Words file counted")
    
    for nf in negativeFiles:
        with open(nf, "r", encoding = 'utf-8')as f:
            line = f.readline()
            counter = len(line.split())
            numWords.append(counter)
    print("Negative Words file counted")
    
    numFiles = len(numWords)
    
    print('The total number of files is', numFiles)
    print('The total number of words in the files is', sum(numWords))
    print('The average number of words in the files is', sum(numWords)/len(numWords))
    print('document with highest number of words', max(numWords))

    return max(numWords), numFiles #[numFiles,max(numWords)]

def load_model(model_name):
    try:
        os.chdir(ROOT + "models/")
        model = ROOT + "models/" + model_name
        
        data = np.load(model)
        os.chdir("..")
        return data        
    
    except Exception as e:
        
        print("Model name {} incorrect. Error - {}".format(model_name,e))
        
    

