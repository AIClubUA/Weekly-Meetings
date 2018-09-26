
import pandas as pd 

file_name = "all_contracts_prime_transactions_1.csv"
relative_path = "../2018_all_Contracts_Full_20180917/all_contracts_prime_transactions_1.csv"
#absolute_path = "C:/Users/Instructor/Desktop/io/"
#print(absolute_path)

"""
import os

os_path = os.getcwd()
full_path = os_path + "\\" + relative_path

print(os_path)
print(full_path)
"""
"""
df = pd.read_csv(relative_path, nrows=100)

#df.to_csv("smaller-data.csv", index=False)

print(df.columns)

small_frame = pd.DataFrame({})

small_frame['transaction_number'] = df['transaction_number']
"""

df = pd.read_csv("smaller-data.csv", nrows=100)


cols_to_keep = [
    "potential_total_value_of_award",
    "award_id_piid",
    "recipient_name",
    "funding_agency_name"
]

for header in df.columns:
    if header not in cols_to_keep:
        df.drop(columns=[header], inplace=True)

#----------------------------------------------------------

def to_list(df):
    return df.values.tolist()

lol = to_list(df)

def to_dict(df):
    return df.to_dict('list')

data = to_dict(df)

print("keys:", data.keys())

print(data["potential_total_value_of_award"])

"""
for i in data['potential_total_value_of_award']:
    print(i)
"""


# list of rows
#lol

# dictionary of columns
#data

import pickle

filename = "list_of_list.dat"

string = "this is a random string that isnt a file but it is a piece of data"


"""
pickle.dump(string, open(filename, 'wb'), -1)
print("saved .dat")
"""
# -------------- Reading In ----------------
# next, we are going to read it in to new variable
data_read_in = pickle.load(open(filename, 'rb'))

print(data_read_in)
