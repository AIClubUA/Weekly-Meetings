
import os
import csv

import numpy as np 
import pandas as pd
import tensorflow as tf 

from datetime import datetime
start = datetime.now()

import spacy
nlp = spacy.load('en')

# User Defined Functions
import backend as bk


# ------------------ Hyper Parameters -----------------------------------
text_seq_word_ct = 300 # Number of words in a sequence length
vec_size = 384 # embedding_dimension

hidden_layer_size = 515 # Number of neurons in the internal layers

num_classes = 6 # Number of classes we are predicting, aka the last layer in the network

# ------------------ Tensor Shapes --------------------------------------
# Tensorflow placehodler variable declaration
_seqlens = tf.placeholder(tf.int32, shape=[None], name='seqlens')

# word embedding imput shape
embed = tf.placeholder(tf.float32, shape=[None, text_seq_word_ct, vec_size], name='embed')

# output dimensions
_labels = tf.placeholder(tf.float32, shape=[None, num_classes], name='labels') # Labels 

# ------------------ LSTM Initialization --------------------------------
# RNN w/ single LSTM cell architecture
with tf.variable_scope("lstm"):
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_layer_size, forget_bias=1.0) # Basic LSTM Cell yo
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, embed, sequence_length=_seqlens, dtype=tf.float32)

# y = mx + b --> y = weights(x) + biases
weights =  tf.Variable(tf.truncated_normal([hidden_layer_size, num_classes], mean=0, stddev=0.01), name="weights")
biases = tf.Variable(tf.truncated_normal([num_classes], mean=0, stddev=0.01), name="biases")

# ------------------ Network Outputs ------------------------------------
# Extract the last relevant output and use in linear layer
final_output = tf.add(tf.matmul(states[1], weights), biases)
pred = tf.nn.sigmoid_cross_entropy_with_logits(logits=final_output, labels=_labels) # Sigmoid allows for multi class, as opposed to softmax being /1
final_output_sig = tf.sigmoid(final_output, name="final_output_sig")
cross_entropy = tf.reduce_mean(pred) # cost, loss defining function

# ------------------ Training and Loss Optimization ---------------------
train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

# ------------------ Predictions and accuracy ---------------------------
correct_prediction = tf.equal(tf.argmax(_labels, 1), tf.argmax(final_output, 1))
accuracy = (tf.reduce_mean(tf.cast(correct_prediction, tf.float32)))*100

# ------------------- Save it for later ---------------------------------
saver = tf.train.Saver() # Once the network is defined, we can save the architecture for later use



#------------------- Create Prediction .csv Structure -----------------------------------------------
# Top row of .csv
headers = ['id', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
outframe = [headers] # List of lists where each row will have values corresponding to each column

batch_size = 16
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, os.getcwd() + "/models/lstm-rnn.ckpt")

    #--------------------------------------------------------------------------------------------------------------------------------
    tdf = pd.read_csv(os.getcwd() + "/data/test.csv")
    names = list(tdf["id"])
    test_x = list(tdf["comment_text"])

    for i in range(0, len(test_x), batch_size):
        print("Pred:", str(i) + '/' + str(len(test_x)))
        in_x, sq_len = bk.get_test(test_x[i:i+batch_size], nlp, text_seq_word_ct, vec_size)
        output_example = sess.run(final_output_sig, feed_dict={embed:in_x, _seqlens:sq_len})
        for j in range(len(output_example)):
            buffer = [names[i+j]]
            buffer.extend(output_example[j])
            outframe.append(buffer)

    bk.csvWriteRow(outframe, "/data/predictions.csv")
