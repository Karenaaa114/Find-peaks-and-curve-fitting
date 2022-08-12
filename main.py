import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from method import *


# when use data "ZnO transformation.csv"  
# def main():
    # filename = "ZnO transformation.csv"
    # data = read_data(filename)
    # filename = "PdCeO2"
    # data = open_file(filename)
    # two_theta, intensity = separate_x_y_axis(data)
    # plot_data(two_theta, intensity)
    # gaussian_fitting_plot(two_theta,intensity[50],[20,35],[[26.4,1,600],[28.4,1,750],[30,1,750]])
    # gaussian_plot_error(two_theta,intensity[60],[20,35],[[26.4,1,600],[28.4,1,750],[30,1,750]])




# when use data "PdCeO2" with all .rg file in the folder
# def main():
#     filename = "PdCeO2"
#     data = open_file(filename)
#     two_theta, intensity = separate_x_y_axis(data)
#     plot_data(two_theta, intensity)
#     #plot every dataset in the folder
#     for i in range(len(intensity)):
#         plt.title('Gaussian fitting for dataset %d' %i)
#         gaussian_fitting_plot(two_theta,intensity[i],[1.8,4.2],[[2.4,0.038,0.3],[3.8,0.07,1.13]])

#     gaussian_plot_error(two_theta,intensity[3],[1.8,4.2],[[2.4,0.038,0.3],[3.8,0.07,1.13]])
#     lorentzian_plot_error(two_theta,intensity[3],[1.8,4.2],[[2.4,0.038,0.3],[3.8,0.07,1.13]])
#     PseudoVoigt_plot_error(two_theta,intensity[3],[1.8,4.2],[[2.4,0.038,0.3],[3.8,0.07,1.13]])
#     toCsv(two_theta,intensity,[1.8,4.2],[[2.4,0.038,0.3],[3.8,0.07,1.13]])




def main():
    data_type = input('Input the data type: csv or multiple files?')
    if data_type == 'csv':
        filename = input('Please input the file name:')
        data = read_data(filename)
        two_theta, intensity = separate_x_y_axis(data)
        # "ZnO transformation.csv"
    elif data_type == 'multiple files':
        # filename = "PdCeO2"
        filename = input('Please input the file name:')
        data = open_file(filename)
        two_theta, intensity = separate_x_y_axis(data)
    else:
        print("Input the data type: csv or multiple files?")

    
    plot = input('Do you want to plot the data?yes or no?')
    if plot == 'yes':
        print(plot_data(two_theta, intensity))
    elif plot == 'no':
        print('That is fine!')
    else:
        print('Do you want to plot the data?yes or no?')

    








if __name__ == "__main__":
    main()