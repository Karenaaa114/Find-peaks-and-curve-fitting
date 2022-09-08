# Find-peaks-and-curve-fitting
## Introduction to this program
This is a multi-peak XRD fitting program developed for multi-file analysis. For peak analysis, the program provides three fitting models: Gaussian, Lorentzian and PseudoVoigt, as they are the commonly used profile functions. This program can analyse multitude of data sets by fitting the peaks observed in X-ray diffraction measurements and determine the information within an individual sets of peaks, specifically the (a) position of the peak, (b) full width at half maximum and (c) area under the peak. 

## Code Function Description
method.py:
1. Read files 
This program can read two different types of data. (a) A .csv file that involved all the data with the first column which is ‘2-theta’ data (x-axis). The rest of the columns are ‘intensities’ data (y-axis) with each column being one intensity data for each data set measured in different temperature or time. (b) all the files with .gr format in one folder. In each .gr file, the first column (x-axis) is real atom distance and the second column (y-axis) is intensity. Read files in numeric order by file name and read column by column. 
2.Plot 2D data
Users can plot the data in 2D for selected range

3.Plot 3D data
Users can plot the data in 3D for selected range

4.generate the baseline
This program generates baseline to do a background subtraction

5.fitting the curve with Gaussian/Lorentzian/PseudoVoigt methods and plot
By input guess of center, sigma and amplitude of peaks and baseline parameters, this program can plot for selected/all data set.

7.plot Gaussian/Lorentzian/PseudoVoigt fitting result
The original data, fitting data and the difference between these two data (error) can be shown in the fitting result graph.

8.print fitting index
This program uses chi-square to calculate the fitting index

9.export information of peaks to excel.
By input guess of center, sigma and amplitude of peaks, .csv files of peaks’ information will be outputted. The csv file contains amplitude, center, sigma, full Width at half Maximum, height and the errors of these parameters. It is worth noting that for every peak there will be a csv file stored in ‘peakFiles’ folder. 

10.plot variation in full width at half maximum(FWHM) and variation in height


main.py:
In this file, users can use two kinds of terminals. 
![image]()


## Platform
This program is written in Python 3.8.8 version and can be run on MAC system. The Integrated Development Environment used in this program is Visual Studio Code. Users can download it to run the program.