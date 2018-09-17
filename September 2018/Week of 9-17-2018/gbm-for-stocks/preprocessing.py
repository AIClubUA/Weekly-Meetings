
import os
import pandas as pd
import yapywrangler as yp
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def loadData(stocks):
    data = {}
    for stock in stocks:
        df = pd.read_csv(os.getcwd() + "/YPData/" + stock + ".csv")
        data[stock] = df.values.tolist()
    
    return data


def group_inputs_and_label(closes):
    close_groups = []
    counter = 0
    buffer = []
    for day in closes:
        buffer.append(day)
        counter += 1
        if counter == 11:
            close_groups.append(buffer)
            buffer = []
            counter = 0

    """
    for i in close_groups:
        print(i)
    """
    return close_groups

def normalize(close_groups):
    normed_data = []

    for group in close_groups:
        norm = preprocessing.normalize([group])
        #print(norm) # see list of list
        normed_data.append(norm[0])

    return normed_data


def inputs_labels_creater(grouped_data):
    inputs = []
    labels = []

    for brick in grouped_data:
        if brick[0]*1.01 > brick[1]:
            labels.append(2)
        elif brick[0]*0.99 < brick[1]:
            labels.append(1)
        else:
            lablels.append(0)
        
        inputs.append(brick[1:])

    print("label:", labels[0], "inputs:", inputs[0])

    return inputs, labels