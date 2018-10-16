
"""
Topics:
    1) Dealing with NA values

    2) Enrichment
        a) Normalization
        b) One hot Encoding
"""

import pandas as pd

"""
PATH = 'dirty_data.csv'
df = pd.read_csv(path)
print(df.isnull().any())

# dictionary of columns name keys, to the value to fill the NaNs with
vals = {
    'Age':0,
    'Cabin':0,
    'Embarked':0,
    # 'some_string':''
}

# To fill NaN in whole dataset we use:
#df.fillna(value=vals, inplace=True)
#df = df.fillna(value=0)

print(df.head(10))

# cannot fill column of integers with strings
string = [i for i in df.int_col]
for s in string:
    if 'ex' in s:
        print("found")
"""

"""
# Now what if we want to fill with the average of a column?
path = "train.csv"
df = pd.read_csv(path)

ages = [i for i in df.Age]

age_sum = 0
age_ct = 0
for age in ages:
    if pd.isna(age):
        #print("!! NA !!")
    else:
        #print(age)
        age_sum += age
        age_ct += 1

avg = age_sum/age_ct
print("Average Age:", avg)

clean_ages = []
for age in ages:
    if pd.isna(age):
        clean_ages.append(avg)
    else:
        clean_ages.append(age)

df['Age'] = clean_ages
"""

"""
# --------------------
# After filling the NaN values, many times we want to Normalize
#norm_val = (val - min)/(max - min)

normed_ages = []
age_min = min(clean_ages)
age_max = max(clean_ages)
for age in clean_ages:
    # norm operation
    norm_val = (age - age_min)/(age_max - age_min)
    normed_ages.append(norm_val)


df['Age'] = normed_ages

print(df.Age)
"""

"""
For categorical values, like the day of the week or the passenger class:
    the number assigned to represent the class holds no information - it is just a name
    we want to convert these non relational numbers into flags
--------------------------------------------
One hot Encoding

Pclasses = 4

[0, 0, 0, 0]

[1, 0, 0, 0] = 1
[0, 1, 0, 0] = 2
[0, 0, 1, 0] = 3
[0, 0, 0, 1] = 4
"""
"""
# Manual implementation using built in python

pclass = [i for i in df.Pclass]

one_hot_pclass = []

for p in pclass:
    
    #buffer = [0, 0, 0]
    #if p == 1:
    #    buffer[0] = 1
    #if p == 2:
    #    buffer[1] = 1
    

    buffer = [0 for i in range(max(pclass))]
    buffer[p-1] = 1
    
    one_hot_pclass.append(buffer)


for item in one_hot_pclass:
    print(item)
"""

"""
# Super clean implementation using pandas

one_hot = pd.get_dummies(df['Pclass'])
print(one_hot.head())

df.drop('Pclass', axis=1, inplace=True)

df = df.join(one_hot)

print(df.head())
"""




