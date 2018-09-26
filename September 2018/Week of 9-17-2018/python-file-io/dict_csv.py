

import pandas as pd 


def writeDictionaryWithPandas(file_name):
    out_data = {
        "vehicle": ["boat", "car", "bike"],
        "weight": [4500, 1500, 20],
        "travel on land": [False, True, True]
    }

    df = pd.DataFrame(out_data)

    # Function we want:
    df.to_csv(file_name, index=False)

file_name = "pandas-outfile.csv"
#writeDictionaryWithPandas(file_name)

df = pd.read_csv(file_name)

print(df.head())

