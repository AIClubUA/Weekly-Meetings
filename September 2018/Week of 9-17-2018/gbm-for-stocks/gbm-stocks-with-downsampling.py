
"""
1)  First things first, we will want to collect the data

If you want to download individual .csv:
    https://finance.yahoo.com/quote/TSLA/history?p=TSLA

If you want to grab a ton at once:
    pip install yapywrangler
"""

import os
#import yapywrangler as yp
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier 

import preprocessing as bk # backend

stocks = ['FB', 'TSLA', 'BAC']

# To download new Data
#data = yp.securityData(stocks, end='2010-01-01', save=True, epoch=False)


"""
2)  Next we want to turn that data into our deisred format
    There are many different ways to do this, to each their own
    Here we will be using basic python data structures
        (pandas is best practice and runs faster, but can be kindof confusing if not in the dataframe headspace)

    The end goal, is chunks of 10 day increments of closing stock price, to predict if tomorrow will go up or down
"""

# To read in existing - can also use pd.read_csv() for each indiv csv.
#data = yp.readExisting(stocks, end='2011-01-01')
data = bk.loadData(stocks)

def visualize_data(data):
    print(type(data)) # dictionary
    print(data.keys()) # see keys = ['BAC', 'FB', 'TSLA'] 

    
    for stock in data.keys():
        #print(type(data[stock])) # see all data is in the format of list
        #print(data[stock]) # see the data format is list of lists
    
        #print(data[stock][0])
        close = data[stock][0][4]
        print(stock, "close:", close)
    

#visualize_data(data)

# --------- Actual Tranformations ----------

bac = data['BAC']
fb = data['FB']
tsla = data['TSLA']

# Remember, just want to get a list of closes to create:
#   inputs = 10 previous days
#   label = tomorrow

bac_closes = [day[4] for day in bac]
fb_closes = [day[4] for day in fb]
tsla_closes = [day[4] for day in tsla]


# b/c doing three times, want to define funciton (if anything is done more than once, it ought to be a function)
bac_grouped = bk.group_inputs_and_label(bac_closes)
fb_grouped = bk.group_inputs_and_label(fb_closes)
tsla_grouped = bk.group_inputs_and_label(tsla_closes)

"""
for i in fb_grouped[:10]:
    print(i)

print("Num samples:", len(fb_grouped))
"""


# Here, our labels are going to be index 0 (most recent day at top of script)
# Next because these symbols all trade at different prices, we want to normalize the chunks

#from sklearn import preprocessing
bac_norm = bk.normalize(bac_grouped)
fb_norm = bk.normalize(fb_grouped)
tsla_norm = bk.normalize(tsla_grouped)

bac_inputs, bac_labels = bk.inputs_labels_creater(bac_norm)
fb_inputs, fb_labels = bk.inputs_labels_creater(fb_norm)
tsla_inputs, tsla_labels = bk.inputs_labels_creater(tsla_norm)

inputs = []
inputs.extend(bac_inputs)
#print("After BAC .extend()", len(inputs))
inputs.extend(fb_inputs)
inputs.extend(tsla_inputs)

labels = []
labels.extend(bac_labels)
labels.extend(fb_labels)
labels.extend(tsla_labels)

print("Num Inputs:", len(inputs))
print("Num Labels:", len(labels))



# Finally, time to split into training and testing sets (for validation)
# from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=.2)
#-----------------------------------------------------------------------------------------------------

"""
3) Time to actually implement the model!

We will be using GBM
    Fast, easy and works pretty dang well
    Great for datasets not large enough to adequatly train a NN on

    http://blog.kaggle.com/2017/01/23/a-kaggle-master-explains-gradient-boosting/
    https://towardsdatascience.com/boosting-algorithm-gbm-97737c63daa3

"""
"""
#from sklearn.ensemble import GradientBoostingClassifier 
gbm = GradientBoostingClassifier( learning_rate=0.01, # [OK]
                                        n_estimators=3600, # [OK]
                                        min_samples_split=10, # [OK]
                                        min_samples_leaf=50, # [OK]
                                        max_depth=14, # [OK]
                                        max_features='sqrt',
                                        subsample=0.9, # .8 [OK]
                                        random_state=10) # [OK]

gbm.fit(X_train, y_train)
score = gbm.score(X_test, y_test)
print("Validation/Testing Score:", score)

test_x = X_test[:10]
test_y = y_test[:10]

preds = gbm.predict(test_x)
"""


"""
4) Review and Improve
    Look for bias
    Look for overfitting
    
Awesome resource for how to tune these and use Grid Searching
    https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/

XGBoost is another very popular technique, basically this one on steroids
    In my experience:
        much harder to implement
        doesnt always out perform
        primary benefit is does processing faster, so you are able to Grid Search in less time

"""

zeros = 0
ones = 0
twos = 0

for i in labels:
    if i == 0:
        zeros += 1
    elif i == 1:
        ones += 1
    elif i == 2:
        twos += 1

print("zeros:", zeros)
print("ones:", ones)
print("twos:", twos)
#-------------------------------------
hacky_x = []
hacky_y = []

ones_ct = 0
twos_ct = 0
for i in range(len(labels)):
    if labels[i] == 1 and ones_ct < 150:
        hacky_x.append(inputs[i])
        hacky_y.append(labels[i])
        ones_ct += 1

    if labels[i] == 2 and twos_ct < 150:
        hacky_x.append(inputs[i])
        hacky_y.append(labels[i])
        twos_ct += 1




X_train, X_test, y_train, y_test = train_test_split(hacky_x, hacky_y, test_size=.2)

#from sklearn.ensemble import GradientBoostingClassifier 
gbm = GradientBoostingClassifier( learning_rate=0.01, # [OK]
                                        n_estimators=3600, # [OK]
                                        min_samples_split=10, # [OK]
                                        min_samples_leaf=50, # [OK]
                                        max_depth=14, # [OK]
                                        max_features='sqrt',
                                        subsample=0.9, # .8 [OK]
                                        random_state=10) # [OK]

gbm.fit(X_train, y_train)
score = gbm.score(X_test, y_test)
print("Validation/Testing Score:", score)

test_x = X_test[:10]
test_y = y_test[:10]

preds = gbm.predict(test_x)

print(test_y)
print(preds)


