# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 14:58:58 2018

@author: Cyrus Dsouza 
"""

import os
import tensorflow as tf
tf.reset_default_graph()




def tf_graph(maxLength):
    batchsize = 50
    lstmUnits = 20
    num_classes = 2
    iterations = 1000
    
    
    labels = tf.placeholder(tf.float32, [batchsize, num_classes])
    input_data = tf.placeholder(tf.int32, [batchsize, maxLength])


if __name__ == "__main__":
    tf_graph(23)
    
