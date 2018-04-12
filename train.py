# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 19:54:59 2018

@author: Cyrus Dsouza
"""

import numpy as np
import re
import os
from os.path import isfile, join

from tools import preprocess
from tools import helper_function as hf
    
import calculate_lengths
import tensorflow as tf
from rnn import TensorFlow

ROOT = "D:/Cyrus/Implementations//Sentiment-Analysis-with-LSTM/"


#get vectors and id's from training data pickle file

#if not available, one would need to create this file, through checking the vocabulary and extracting vectors from GLOVE or gensim's pretrained models.
wordsList = np.load(ROOT + 'training_data/training_data/wordsList.npy')
wordsList = wordsList.tolist()
wordsList = [words.decode('UTF-8') for words in wordsList]
wordVectors = np.load(ROOT + 'training_data/training_data/wordVectors.npy')


    
def convert2id(f_counter,id_matrix,sentence):
    #create lookup function using tensorflow
    for index,i in enumerate(sentence):
        try:
            id_matrix[f_counter][index] = wordsList.index(i)
        except ValueError :
            id_matrix[f_counter][index] = 999999
        
    return id_matrix

    
def parser(id_matrix,files):
    for f_counter,file in enumerate(files):
        with open(file,'r') as f:
            line = f.readline()
            cleanedLine = preprocess.clean(line)
            splitLine = cleanedLine.split()
            convert2id(f_counter,id_matrix, splitLine)
            
    return id_matrix
                
            
    
if __name__ == "__main__":
    
    train = False 

            
#    read the files (currently only 10)
#    print("Reading Files....\n")
#    positiveFiles = [ROOT + 'training_data/training_data/positiveReviews/' + f for f in os.listdir(ROOT + 'training_data/training_data/positiveReviews/') if isfile((join(ROOT + 'training_data/training_data/positiveReviews/', f)))][:10]
#    negativeFiles = [ROOT + 'training_data/training_data/positiveReviews/' + f for f in os.listdir(ROOT + 'training_data/training_data/positiveReviews/') if isfile((join(ROOT + 'training_data/training_data/positiveReviews/', f)))][:10]
#    
#    print("Calculating Lengths....\n")
#    maxLength, numFiles = calculate_lengths.find_max_length(positiveFiles, negativeFiles)#len(sentence)   
#    id_matrix = np.zeros((numFiles,maxLength), dtype= 'int32')


    if train:      
        print("Creating ID Matrix...")

        
        #positive files 
        id_mat = parser(id_matrix,positiveFiles)
        
        #negative files
        id_mat = parser(id_mat,negativeFiles)
        
        np.save('models/idsMatrix', id_mat)
    
    
    else:
        print("Loading ID Matrix...")
        id_matrix = hf.load_model('id_matrix.npy')
        if len(id_matrix.shape) > 0:
            print("Matrix Loaded")
            print("Displaying first 5 rows: \n", id_matrix[:5], "etc.....")
            
    maxLength = 436 #change later and make dynamic
    
    
    
    a = TensorFlow()
    a.run(id_matrix, maxLength,wordVectors)
#        
#        sess.run(ls.optimizer, {input_data: nextBatch, labels: nextBatchLabels})
#       
#        #Write summary to Tensorboard
#        if (i % 50 == 0):
#            summary = sess.run(merged, {input_data: nextBatch, labels: nextBatchLabels})
#            writer.add_summary(summary, i)
#    
#        #Save the network every 10,000 training iterations
#        if (i % 10000 == 0 and i != 0):
#            save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step=i)
#            print("saved to %s" % save_path)
#        writer.close()
#        
#    iterations = 10
#    for i in range(iterations):
#        nextBatch, nextBatchLabels = hf.getTestBatch();
#        print("Accuracy for this batch:", (sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels})) * 100)        
#    
    
        
        
