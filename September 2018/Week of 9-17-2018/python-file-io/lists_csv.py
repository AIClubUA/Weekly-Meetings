import csv

out_list = [
    ["vehicle", "weight", "travel on land"], # header
    ["boat", 4500, False],
    ["car", 1500, True],
    ["bike", 20, True]
]


def writeRowsofRows(file_name, list_of_lists):

    with open(file_name, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for line in list_of_lists:
            writer.writerow(line)



file_name = "row-outfile.csv"

writeRowsofRows(file_name, out_list)


