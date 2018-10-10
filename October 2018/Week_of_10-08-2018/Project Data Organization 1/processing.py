
import pandas as pd 

# 1) Read in the .csv
path = 'C:\\Users\\Instructor\\Desktop\\meeting\\leagueoflegends\\matchinfo.csv'
path = 'C:/Users/Instructor/Desktop/meeting/leagueoflegends/matchinfo.csv'
path = 'leagueoflegends/matchinfo.csv'

df = pd.read_csv(path, nrows=100) # does the actual reading in
#                         nrows means number of rows to read in (to speed up development)

# 2) Visualization
#print(df.head(10)) # here displays 10 rows

# to look at columns
cols = df.columns # is a list

"""
print(cols)
for column in cols:
    print(column)
"""

# 3) Start selecting data
#  to select a single row
labels = df['bResult']
labels = df.bResult      # both of these are equivalent

# Remove laels from the training dataset
#df.drop(['bResult', 'rResult'], axis=1, inplace=True) # axis=1 -> down the dataframe
inputs = df.drop(['bResult', 'rResult'], axis=1)   # both of these are equivalent
#print(inputs.columns)


# 4) Print each row in the dataframe
"""
for index, row in inputs.iterrows():
    print("------------------------------")
    print(row['Season'], row['gamelength'], row['redTopChamp'])
    #print(row)
"""

#------------------------------------
columns = []
for col in inputs.columns:
    columns.append(col)

#print("Multi line:")
#print(columns)

#               v operation before the for loop
easy_columns = [col for col in inputs.columns] #        Both of these are equivalent
#print("Single line:")
#print(easy_columns) 
#---------------------------------------------

"""
mega_list = []
mega_list.append(easy_columns)
"""

mega_list = [easy_columns]
#print(mega_list)

# Convert to list of lists
inputs_list = inputs.values.tolist()

# Add each row to mega list so the headers are there with the data
for row in inputs_list:
    mega_list.append(row)

gl_sum = 0
for row in mega_list[1:11]:
    #gl_sum = gl_sum + row[6]
    gl_sum += row[6]    # these are equivalent
    #print(row)

# ----------------------------------------------------------------------------
# Dictionary Operations

# transforms from dataframe to dictionary
inputs_dict = inputs.to_dict('list')

# What the keys are
print(inputs_dict.keys())

# Iterating dataframes
# 1) By key
"""
for key in inputs_dict.keys():
    print(key, inputs_dict[key])
"""

# 2) By items()
"""
for key, value in inputs_dict.items():
    print(key, value)
"""

gl_sum = sum(inputs_dict['gamelength'])

new_gl_x4 = [i * 4 for i in inputs_dict['gamelength']]

inputs_dict['4xgl'] = new_gl_x4

print(inputs_dict.keys())
