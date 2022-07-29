import csv
import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import find_peaks
from scipy.optimize import curve_fit

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

# two_theta, intensity = separate_x_y(data)
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

# log10 of all data
# two_theta_log = np.log10(two_theta)
# intensity_log = np.log10(intensity)

def plot_data(two_theta, intensity):
    """Plot the graph of data.

    Args:
        two_theta (1-D array)
        intensity (2-D array)
    """
    for i in range(intensity.shape[0]):
        plt.plot(two_theta, intensity[i], linewidth = 0.5)
    plt.title("intensity")
    plt.xlabel(r'$2\theta$')
    plt.ylabel("intensity")
    plt.show()

def plot_data_log10(two_theta, intensity):
    """Plot the graph of data after log10.

    Args:
        two_theta (1-D array)
        intensity (2-D array)
    """
    for i in range(intensity.shape[0]):
        plt.loglog(two_theta, intensity[i], linewidth = 0.5)
        # plt.plot(two_theta_log, intensity_log[i], linewidth = 0.5)
    plt.title("intensity")
    plt.xlabel(r'$2\theta$')
    plt.ylabel("intensity")
    plt.show()

def get_index(datax,x_value):
    """Get index of the x value.

    Args:
        datax (1-D array)
        x_value (2-D list)

    Returns:
        lo: index of the x value
    """
    lo = 0
    for lo in range(len(datax)):
        if datax[lo] >= x_value:
            break
    return lo

def getPeak(datax,datay,x_interval):
    """

    Args:
        datax (_type_): _description_
        datay (_type_): _description_
        x_interval (_type_): _description_

    Returns:
        _type_: _description_
    """
    min_index = get_index(datax,x_interval[0])
    max_index = get_index(datax,x_interval[1])
    y = datay[min_index:max_index+1]
    plt.plot(datax[min_index:max_index+1],y)
    return find_peaks(y,height=0,distance=100)[0]+min_index

def getAllPeaks(datax,datay,x_interval):
    """_summary_

    Args:
        datax (_type_): _description_
        datay (_type_): _description_
        x_interval (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Peaks = []
    for intens in datay:
        peak = getPeak(datax,intens,x_interval)
        if len(peak.tolist()) > 0:
            peakindex = peak.tolist()[0]
            Peaks = [datax[peakindex],intens[peakindex]]
            plt.scatter([datax[peakindex]],[intens[peakindex]],marker='x')
            print(Peaks)
        else:
            print("There is no peak")
        # Peaks.append([datax[pV],intens[pV]])
    return Peaks