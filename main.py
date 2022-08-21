import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from method import *


#when use data "PdCeO2" with all .rg file in the folder
# def main():
#     filename = "PdCeO2"
#     two_theta, intensity = open_gr_file(filename)
#     plot_data(two_theta, intensity)
#     plot_data_3d(two_theta, intensity)
#     dataset_number = 3
#     x_interval = [1.8,4.2]
#     set_pars = [[2.4,0.038,0.3],[3.8,0.07,1.13]]
#     baseline_pars = [10000,0.01]
#     # print(gaussian_fit_index(two_theta,intensity[dataset_number],x_interval,set_pars,baseline_pars))
#     # gaussian_fitting_plot_all(two_theta,intensity,x_interval,set_pars,baseline_pars)
#     # lorentzian_fitting_plot_all(two_theta,intensity,x_interval,set_pars,baseline_pars)
#     # PseudoVoigt_fitting_plot_all(two_theta,intensity,x_interval,set_pars,baseline_pars)

#     # gaussian_plot_error(two_theta,intensity[dataset_number],x_interval,set_pars,baseline_pars)
#     # lorentzian_plot_error(two_theta,intensity[dataset_number],x_interval,set_pars,baseline_pars)
#     # PseudoVoigt_plot_error(two_theta,intensity[dataset_number],x_interval,set_pars,baseline_pars)
#     toCsv(two_theta,intensity,x_interval,set_pars,baseline_pars)
#     all_change_fwhm()
#     all_change_height()


# when use data "ZnO transformation.csv" 
def main():
    filename = "ZnO transformation.csv"
    two_theta, intensity = read_csv_file(filename)
    # plot_data(two_theta, intensity)
    # plot_data_3d(two_theta, intensity)
    # plot_data_3d_range(two_theta, intensity, [46.5,48.5])
    # plot_data_3d_range(two_theta, intensity, [24,34])
    dataset_number = 3
    # x_interval = [20,35]
    # set_pars = [[26.4,1,600],[28.4,1,750],[30,1,750]]
    # baseline_pars = [10000,0.0001]
    # x_interval = [27,29.5] #（002）
    # set_pars = [[28.4,1,940],[0,0,0]]
    # baseline_pars = [10000,0.0001,1250]
    x_interval = [42,46.8] #（022）
    set_pars = [[44.8,2,800],[0,0,0]]
    baseline_pars = [10000,0.0001,-1]
    # x_interval = [46.5,48.5] #（110）
    # set_pars = [[47.3,1,500],[0,0,0]]
    # baseline_pars = [10000,0.0001,500]
    # print(gaussian_fit_index(two_theta,intensity[dataset_number],x_interval,set_pars,baseline_pars))
    # gaussian_fitting_plot(two_theta,intensity[60],x_interval,set_pars,baseline_pars)
    # gaussian_fitting_plot_all(two_theta,intensity,x_interval,set_pars,baseline_pars)
    # lorentzian_fitting_plot_all(two_theta,intensity,x_interval,set_pars,baseline_pars)
    # PseudoVoigt_fitting_plot_all(two_theta,intensity,x_interval,set_pars,baseline_pars)

    # gaussian_plot_error(two_theta,intensity[dataset_number],x_interval,set_pars,baseline_pars)
    # lorentzian_plot_error(two_theta,intensity[dataset_number],x_interval,set_pars,baseline_pars)
    # PseudoVoigt_plot_error(two_theta,intensity[dataset_number],x_interval,set_pars,baseline_pars)
    toCsv(two_theta,intensity,x_interval,set_pars,baseline_pars)
    # all_change_fwhm()
    all_change_height()
    # change_height("peak2.csv")





# def main():
#     data_type = input('Input the data type: csv or multiple files?')
#     if data_type == 'csv':
#         filename = input('Please input the file name:')
#         two_theta, intensity = read_csv_file(filename)
#         # "ZnO transformation.csv"
#     elif data_type == 'multiple files':
#         # filename = "PdCeO2"
#         filename = input('Please input the file name:')
#         two_theta, intensity = open_gr_file(filename)
#     else:
#         print("Input the data type: csv or multiple files?")

    
#     plot = input('Do you want to plot the data?yes or no?')
#     if plot == 'yes':
#         plot_data(two_theta, intensity)
#     elif plot == 'no':
#         print('That is fine!')
#     else:
#         print('Do you want to plot the data?yes or no?')


#     plot_3d = input('Do you want to plot 3D version of the data?yes or no?')
#     if plot_3d == 'yes':
#         plot_data_3d(two_theta, intensity)
#     elif plot_3d == 'no':
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

#     baseline_parameter = input('what are the smoothness and asymmetry guess for baseline?')
#     base_parameter = baseline_parameter.split()
#     # print('list: ', interval)
#     for i in range(len(base_parameter)): 
#         base_parameter[i] = float(base_parameter[i]) 


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
#     gaussian_fitting_plot(two_theta,intensity[number],interval,set_all_pars,baseline_parameter)
#     gaussian_plot_error(two_theta,intensity[number],interval,set_all_pars,baseline_parameter)
#     lorentzian_plot_error(two_theta,intensity[number],interval,set_all_pars,baseline_parameter)
#     PseudoVoigt_plot_error(two_theta,intensity[number],interval,set_all_pars,baseline_parameter)


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