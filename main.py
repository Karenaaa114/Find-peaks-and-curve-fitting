import csv
import matplotlib.pyplot as plt
import numpy as np

# filename = "NH4OH-FAU-Practice-data.csv"
# data = read_data(filename)
def read_data(filename):
    """Return the data from csv file and transpose the data to read by row.

    Args:
        filename (.csv file)

    Returns:
        data_transposed: data
    """

    data = []
    with open(filename, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)  
        for column in csv_reader:            
            data.append(column)

    data_transposed = np.array(data).transpose()
    data_transposed = data_transposed.astype(np.float32)
    return data_transposed 

def separate_x_y_axis(data):
    """To separate the data into two-theta(x-axis) and intensities(y-axis).

    Args:
        data (2-D array): the first line is two_theta(x-axis) and others are intensity(y-axis).

    Returns:
        two_theta, intensity: two theta(x-axis) and intensities in different temperature(y-axis).
    """
    two_theta = data[0]
    intensity = data[1:]
    return two_theta, intensity