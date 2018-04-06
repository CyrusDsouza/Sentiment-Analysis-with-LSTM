# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 11:57:18 2018

@author: CY291970
"""
import os
import numpy as np 
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt

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

#plt.hist(np.array(numWords),25)
#plt.xlabel('Sequence Length')
#plt.ylabel('Frequency')
#plt.axis([0, 450, 0, 10])
#plt.show()

if __name__ == "__main__":
    find_max_length()
    