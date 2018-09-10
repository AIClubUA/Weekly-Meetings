
import os
import csv

import numpy as np 
import pandas as pd
import tensorflow as tf 

from datetime import datetime
start = datetime.now()

import spacy
nlp = spacy.load('en')


def toxicCommentsOnly(train_x, train_y):
    ones_x = []
    ones_y = []
    for i in range(len(train_x)):
        if 1 in train_y[i]:
            ones_x.append(train_x[i])
            ones_y.append(train_y[i])
            
    return ones_x, ones_y


def meaterizer(train_x, nlp, post_size=300, vec_size=384):
    frame = []
    seqlens = []
    for i in train_x:
        buff = []
        doc = nlp(i)

        if len(doc) == post_size: 
            #print("Even")
            seqlens.append(post_size)
            
            for word in doc:
                buff.append(word.vector)
            

        elif len(doc) > post_size:
            #print("Long")
            seqlens.append(post_size)
            bk = post_size/2
            condenser = []
            cond = np.zeros(vec_size)
            for word in doc[0:bk]:
                buff.append(word.vector)
            for word in doc[bk+1:-bk]: # could optimize no doubt
                condenser.append(word)
            for word in condenser:
                #print(len(word.vector))
                cond = np.add(cond, word.vector)
            buff.append(cond)
            for word in doc[-(bk-1):]:
                buff.append(word.vector)


        elif len(doc) < post_size:
            #print("Short")
            seqlens.append(len(doc))

            for word in doc:
                buff.append(word.vector)

            while len(buff) < post_size:
                buff.append(np.zeros(vec_size))
            
        frame.append(buff)

    return frame, seqlens

def get_random_batch(batch_size, data_x, data_y, nlp, post_size, vec_size):
    #print("Fre$h batch")
    instance_indecies = list(range(len(data_x)))
    np.random.shuffle(instance_indecies)
    batch = instance_indecies[:batch_size] 

    x, seqlens = meaterizer([data_x[i] for i in batch], nlp, post_size, vec_size)
    y = [data_y[i] for i in batch]
    #print("Successfully made it.")
    return np.nan_to_num(np.array(x)), np.nan_to_num(np.array(y)), np.nan_to_num(np.array(seqlens))

def get_test(data, nlp, post_size, vec_size):
    x, seqlens = meaterizer(data, nlp, post_size, vec_size)
    return np.nan_to_num(np.array(x)), np.nan_to_num(np.array(seqlens))

def csvWriteRow(yuuge_list, filename):
    filename = os.getcwd() + filename
    
    if '.csv' not in filename:
        filename = filename + '.csv'

    with open(filename, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for line in yuuge_list:
            writer.writerow(line)
    
    print('[+] Successfully exported data to', filename, '[+]\n')


