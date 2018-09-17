


headers = ["first", "last", "major"]
row_1 = ["lee", "skinner", "mechanical"]
row_2 = ["kyle", "smith", "news media"]
row_3 = ["ben", "robertson", "aerospace"]

#print(headers)
#print(row_1)
#print(row_2)
#print(row_3)


sheet = [
    headers,
    row_1,
    row_2,
    row_3
]

for row in sheet:
    print(row)


import csv
row_out = "spreadsheet.csv"

list_of_lists = sheet

with open(row_out, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    for line in list_of_lists:
        writer.writerow(line)
