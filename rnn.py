# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 14:58:58 2018

@author: Cyrus Dsouza 
"""

import os
import tensorflow as tf
tf.reset_default_graph()

from tools import helper_function as hf

class TensorFlow(object):
    def __init__(self):
        pass
    

    def tf_graph(self,maxLength,wordVectors):
        print("Creating TensorGraph...")
        self.batchsize = 50
        self.lstmUnits = 20
        self.num_classes = 2
        self.iterations = 1000
        self.dimension = 300
        
        self.labels = tf.placeholder(tf.float32, [self.batchsize, self.num_classes])
        input_data = tf.placeholder(tf.int32, [self.batchsize, maxLength])
        
        print("Preparing lookup matrix")
        self.data = tf.Variable(tf.zeros([self.batchsize, maxLength, self.dimension]), dtype= tf.float32)
        self.data = tf.nn.embedding_lookup(wordVectors,input_data)
        
        
        print("Adding lstm cells...")
        lstmCell = tf.contrib.rnn.BasicLSTMCell(self.lstmUnits)
        lstmCell = tf.contrib.rnn.DropoutWrapper(cell = lstmCell, output_keep_prob = 0.75)
        value, _ = tf.nn.dynamic_rnn(lstmCell, self.data, dtype = tf.float32)
        
        weight = tf.Variable(tf.truncated_normal([self.lstmUnits, self.num_classes]))
        bias = tf.Variable(tf.Variable(tf.constant(0.1,shape = [self.num_classes])))
        value = tf.transpose(value, [1,0,2])
        last = tf.gather(value, int(value.get_shape()[0])-1)
        prediction = (tf.matmul(last, weight)+ bias)
        
        correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(self.labels,1))
        accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
        
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = self.labels))
        optimizer = tf.train.AdamOptimizer().minimize(loss)
        
        print("Graph created///")
        
        
        
    def run(self,ids,maxLength,wordvectors):
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        self.tf_graph(maxLength,wordvectors)
        saver = tf.train.Saver()

        iterations = 100000
        for i in range(iterations):
            #Next Batch of reviews
            print(self.batchsize, maxLength)
            nextBatch, nextBatchLabels = hf.getTrainBatch(ids, self.batchsize, maxLength)
            sess.run(optimizer, {input_data: nextBatch, self.labels: nextBatchLabels, wordVectors : wordvectors})
           
            #Write summary to Tensorboard
            if (i % 50 == 0):
                summary = sess.run(merged, {input_data: nextBatch, labels: nextBatchLabels})
                writer.add_summary(summary, i)
        
            #Save the network every 10,000 training iterations
            if (i % 10000 == 0 and i != 0):
                save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step=i)
                print("saved to %s" % save_path)
        writer.close()
            
        iterations = 10
        for i in range(iterations):
            nextBatch, nextBatchLabels = hf.getTestBatch();
            print("Accuracy for this batch:", (sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels})) * 100) 
        import datetime
        
        tf.summary.scalar('Loss', loss)
        tf.summary.scalar('Accuracy', accuracy)
        merged = tf.summary.merge_all()
        logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
        writer = tf.summary.FileWriter(logdir, sess.graph)

if __name__ == "__main__":
    f = TensorFlow()
    f.run()
    
