import csv
import matplotlib.pyplot as plt
import numpy as np

import math

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

def get_index_in_interval(datax,x_interval):
    """Get index of x interval.

    Args:
        datax (1-D array)
        x_interval (1-D list)

    Returns:
        index: index of x interval
    """
    index = np.where( (datax>=x_interval[0]) & (datax<x_interval[1]))[0]
    return index

def getPeak(datax,datay,x_interval):
    """_summary_

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
    peaks = find_peaks(y,height=0,distance=100)[0]+min_index
    return peaks

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

""" input interval return peaks position and graph of the peaks."""
# x_interval = [[1.6,1.9],[6,7]]
x_interval = [[6,7]]
# x_interval = [[1.6,1.9]]
# all_peaks = []
for interval in x_interval:
    print(f"in interval {interval}, the peaks are")
    R = getAllPeaks(two_theta,intensity,interval)
plt.title("intensity")
plt.xlabel(r'$2\theta$')
plt.ylabel("intensity")
plt.show()
    
    # all_peaks.append(R)
# print(all_peaks)
# print(f"peaks:{R} ")

"""define Gaussian function (not sure use whith one)"""
def Gaussian(x,amp,mu,sigma):
    return amp / (sigma * math.sqrt(2 * math.pi)) * np.exp(-(x-mu)**2 / (2*sigma**2))
# def Gaussian(x,amp,mu,sigma):
#     return amp * np.exp(-(x-mu)**2 / (2*sigma**2))

"""adding up N Gaussian distributions"""
def GaussianN(x,parameters):
    g = np.zeros(x.shape[0])
    for para in parameters:
        g = g + Gaussian(x,para[0],para[1],para[2])
    return g
