
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


# -------------------------------------------------- Loading Training Data ---------------------------------------------------
print("[+] Loading Data...")

# Read in data from train.csv
df = pd.read_csv(os.getcwd() + "/data/train.csv")

# Select columns we want to format into the Inputs and corresponding Labels
train_x = list(df["comment_text"]) # Inputs
train_y = list(zip(df["toxic"], df["severe_toxic"], df["obscene"], df["threat"], df["insult"], df["identity_hate"])) # Labels

# Remove original data from RAM (not necessary)
del df 

# To ensure high toxicity, want to create new subset of just comments that are toxic
ones_x, ones_y = bk.toxicCommentsOnly(train_x, train_y)

print("[+] Loading Data Complete [+]")


#----------------------------------------  Network Architecture --------------------------------------------------------------

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

#---------------------------------------------- Executing the Training Graph ---------------------------------------------------------------

batch_size = 128
epochs = 100
num_steps = int((len(ones_x) * epochs) / batch_size) # 3 epochs
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(num_steps): # had [1, 0, 0, 0, 0, 0] figured out around 200
        x_batch, y_batch, seqlen_batch = bk.get_random_batch(batch_size, ones_x, ones_y, nlp, text_seq_word_ct, vec_size)
        sess.run(train_step, feed_dict={embed:x_batch, _labels:y_batch, _seqlens:seqlen_batch})
        if step % 10 == 0:
            acc, aut = sess.run([accuracy, final_output_sig], feed_dict={embed:x_batch, _labels:y_batch, _seqlens:seqlen_batch})
            
            for i in range(5):
                print(y_batch[i], aut[i])
            
            print(str(step) + '/' + str(num_steps) + " R-init RAN: Accuracy at %d: %.5f" % (step, acc))
        
        if step % 10 == 0:
            save_path = saver.save(sess, os.getcwd() + "/models/lstm-rnn.ckpt")

