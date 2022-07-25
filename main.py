import csv
import matplotlib.pyplot as plt
import numpy as np

# filename = "NH4OH-FAU-Practice-data.csv"
def read_data(filename):
    """Return the data from csvfile and transpose the data to read by row.
       Input: csv file
    """
    data = []
    with open(filename, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)  
        for column in csv_reader:            
            data.append(column)

    data_transposed = np.array(data).transpose()
    data_transposed = data_transposed.astype(np.float32)
    return data_transposed 