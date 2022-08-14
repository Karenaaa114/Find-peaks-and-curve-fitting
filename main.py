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
def main():
    filename = "PdCeO2"
    data = open_file(filename)
    two_theta, intensity = separate_x_y_axis(data)
    plot_data(two_theta, intensity)
    #plot every dataset in the folder
    # for i in range(len(intensity)):
    #     plt.title('Gaussian fitting for dataset %d' %i)
    #     gaussian_fitting_plot(two_theta,intensity[i],[1.8,4.2],[[2.4,0.038,0.3],[3.8,0.07,1.13]])

    gaussian_plot_error(two_theta,intensity[3],[1.8,4.2],[[2.4,0.038,0.3],[3.8,0.07,1.13]])
    lorentzian_plot_error(two_theta,intensity[3],[1.8,4.2],[[2.4,0.038,0.3],[3.8,0.07,1.13]])
    PseudoVoigt_plot_error(two_theta,intensity[3],[1.8,4.2],[[2.4,0.038,0.3],[3.8,0.07,1.13]])
    toCsv(two_theta,intensity,[1.8,4.2],[[2.4,0.038,0.3],[3.8,0.07,1.13]])
    all_change_fwhm()
    all_change_height()




# def main():
#     data_type = input('Input the data type: csv or multiple files?')
#     if data_type == 'csv':
#         filename = input('Please input the file name:')
#         data = read_data(filename)
#         two_theta, intensity = separate_x_y_axis(data)
#         # "ZnO transformation.csv"
#     elif data_type == 'multiple files':
#         # filename = "PdCeO2"
#         filename = input('Please input the file name:')
#         data = open_file(filename)
#         two_theta, intensity = separate_x_y_axis(data)
#     else:
#         print("Input the data type: csv or multiple files?")

    
#     plot = input('Do you want to plot the data?yes or no?')
#     if plot == 'yes':
#         print(plot_data(two_theta, intensity))
#     elif plot == 'no':
#         print('That is fine!')
#     else:
#         print('Do you want to plot the data?yes or no?')

    
#     interval_string = input('What range are you planning to fit?')   #[1.8,4.2] in "PdCeO2" file
#     interval = interval_string.split()
#     # print('list: ', interval)
#     for i in range(len(interval)): 
#         interval[i] = float(interval[i]) 

    
#     number = input('Which dataset are you planning to fit?')
#     number = int(number)


#     peaks = input('How many peaks in this dataset?')   #[3]
#     set_all_pars = []
#     for j in range(int(peaks)):
#         pars = []
#         set_pars_string = input('What is your initial guess?')   # [[26,1,600],[28,1,750],[30,1,750]]
#         set_pars = set_pars_string.split()
#         for i in range(len(set_pars)):
#             # set_pars[i] = int(set_pars[i])
#             set_pars[i] = float(set_pars[i])
#             pars.append(set_pars)
#         set_all_pars.append(set_pars)
#     # print(set_all_pars)
# # [3],[1.8,4.2],[[2.35,0.038,0.3],[3.8,0.07,1.13]]

# #plot gaussian, lorentzian, PseudoVoigt fitting result
#     gaussian_fitting_plot(two_theta,intensity[number],interval,set_all_pars)
#     gaussian_plot_error(two_theta,intensity[number],interval,set_all_pars)
#     lorentzian_plot_error(two_theta,intensity[number],interval,set_all_pars)
#     PseudoVoigt_plot_error(two_theta,intensity[number],interval,set_all_pars)


# #plot all of the dataset(one dataset per graph)
#     all_fitting = input('Do you want to plot all of the dataset?yes or no?')

#     if all_fitting == 'yes':
#         for i in range(len(intensity)):
#             plt.title('Gaussian fitting for dataset %d' %i)
#             gaussian_fitting_plot(two_theta,intensity[i],[1.8,4.2],[[2.4,0.038,0.3],[3.8,0.07,1.13]])
#     elif all_fitting == 'no':
#         print('That is fine!')
#     else:
#         print('Do you want to plot all of the dataset?yes or no?')



#     toCsv(two_theta,intensity,interval,set_all_pars)


#     change_fwhm = input('Do you want to see the change in FWHM? yes or no?')
#     if change_fwhm == 'yes':
#         all_change_fwhm()
#     elif change_fwhm == 'no':
#         print('That is fine!')
#     else:
#         print('Do you want to see the change in FWHM? yes or no?')


#     change_height = input('Do you want to see the change in height? yes or no?')
#     if change_height == 'yes':
#         all_change_height()
#     elif change_height == 'no':
#         print('That is fine!')
#     else:
#         print('Do you want to see the change in height? yes or no?')

    








if __name__ == "__main__":
    main()