import pandas as pd

df = pd.read_csv("train.csv")
df.fillna(0, inplace=True)

print(df.head())

"""
predict death or life (1 or 0)

GOAL: process data so that we have our inputs in a logical order on a per person basic
"""

labels = df['Survived']

features = list(df.columns)
features.remove("Survived") # removes "Survived" from index 1

#print(features)

# this is a dataframe pointing to the 'df' variable
inputs = df[features]

# this is a dictionary memory independant
inputs = inputs.to_dict('list')

age = inputs["Age"]

agesum = 0
count = 0

for item in age:
    print(item)

    if item > 0:
        #agesum = agesum + item
        agesum += item
        count += 1

avg = int(agesum / count)

print("avg:", avg)

newage = []
for item in age:
    if item > 0:
        newage.append(int(item))
    else:
        newage.append(avg)
print()
print(" original:", age[:10])
print("corrected:", newage[:10])

inputs["Age"] = newage

print("from inputs:", inputs["Age"][:10])




