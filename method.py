import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import math

from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from lmfit import models
from lmfit.models import GaussianModel, LorentzianModel, PseudoVoigtModel, ExponentialModel

from scipy import sparse
from scipy.sparse.linalg import spsolve


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
    plt.savefig('graph of all dataset')
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

# def get_index(datax,x_value):
#     """Get index of the x value.

#     Args:
#         datax (1-D array)
#         x_value (2-D list)

#     Returns:
#         lo: index of the x value
#     """
#     lo = 0
#     for lo in range(len(datax)):
#         if datax[lo] >= x_value:
#             break
#     return lo

# def get_index_in_interval(datax,x_interval):
#     """Get index of x interval.

#     Args:
#         datax (1-D array)
#         x_interval (1-D list)

#     Returns:
#         index: index of x interval
#     """
#     index = np.where( (datax>=x_interval[0]) & (datax<x_interval[1]))[0]
#     return index

# def getPeak(datax,datay,x_interval):
#     """_summary_

#     Args:
#         datax (_type_): _description_
#         datay (_type_): _description_
#         x_interval (_type_): _description_

#     Returns:
#         _type_: _description_
#     """
#     min_index = get_index(datax,x_interval[0])
#     max_index = get_index(datax,x_interval[1])
#     y = datay[min_index:max_index+1]
#     plt.plot(datax[min_index:max_index+1],y)
#     peaks = find_peaks(y,height=0,distance=100)[0]+min_index
#     return peaks

# def getAllPeaks(datax,datay,x_interval):
#     """_summary_

#     Args:
#         datax (_type_): _description_
#         datay (_type_): _description_
#         x_interval (_type_): _description_

#     Returns:
#         _type_: _description_
#     """
#     # Peaks = []
#     for intens in datay:
#         peak = getPeak(datax,intens,x_interval)
#         if len(peak.tolist()) > 0:
#             peakindex = peak.tolist()[0]
#             Peaks = [datax[peakindex],intens[peakindex]]
#             plt.scatter([datax[peakindex]],[intens[peakindex]],marker='x')
#             print(Peaks)
#         else:
#             print("There is no peak")
#         # Peaks.append([datax[pV],intens[pV]])
#     return Peaks

# """ input interval return peaks position and graph of the peaks."""
# # x_interval = [[1.6,1.9],[6,7]]
# x_interval = [[6,7]]
# # x_interval = [[1.6,1.9]]
# # all_peaks = []
# for interval in x_interval:
#     print(f"in interval {interval}, the peaks are")
#     R = getAllPeaks(two_theta,intensity,interval)
# plt.title("intensity")
# plt.xlabel(r'$2\theta$')
# plt.ylabel("intensity")
# plt.show()
    
#     # all_peaks.append(R)
# # print(all_peaks)
# # print(f"peaks:{R} ")

# """define Gaussian function (not sure use whith one)"""
# def Gaussian(x,amp,mu,sigma):
#     return amp / (sigma * math.sqrt(2 * math.pi)) * np.exp(-(x-mu)**2 / (2*sigma**2))
# # def Gaussian(x,amp,mu,sigma):
# #     return amp * np.exp(-(x-mu)**2 / (2*sigma**2))

# """adding up N Gaussian distributions"""
# def GaussianN(x,parameters):
#     g = np.zeros(x.shape[0])
#     for para in parameters:
#         g = g + Gaussian(x,para[0],para[1],para[2])
#     return g


# """ Gaussian fitting menthod 1 (fitting display unwell)"""
# popt_gauss = []
# pcov_gauss = []
# perr_gauss = []
# for i in range(intensity.shape[0]):
#     interval_index = get_index_in_interval(two_theta, [6,7])
#     select_data_x = two_theta[interval_index]
#     select_data_y = intensity[i][interval_index]
#     popt, pcov = curve_fit(Gaussian, select_data_x, select_data_y, maxfev = 10000)
#     perr = np.sqrt(np.diag(pcov)) #error
#     popt_gauss.append(popt)
#     pcov_gauss.append(pcov)
#     perr_gauss.append(perr)
#     # print(f"popt:{popt} ")
#     # print(f"perr:{perr} ")
#     # xfit = np.linspace(select_data_x.min(),select_data_x.max(),100)
#     xfit = np.linspace(select_data_x[0],select_data_x.max(),100)
#     yfit = Gaussian(xfit,*popt)
#     plt.plot(xfit,yfit,'--',label='fitting')
#     # print(a,center,width)
#     # x_interval = [[6,7]]
# plt.xlim(5.75,7)
# plt.ylim(0,0.02)
# plt.title("Gaussian fitting")
# plt.xlabel(r'$2\theta$')
# plt.ylabel("intensity")
# # plt.legend()
# plt.show()

# """print height, center, width and area for Gaussian fitting"""
# for i in range(len(popt_gauss)):
#     print(f"\nthe {i}th line: ")
#     print( "height = %0.7f (+/-) %0.7f" % (popt_gauss[i][0], perr_gauss[i][0]))
#     print( "center = %0.7f (+/-) %0.7f" % (popt_gauss[i][1], perr_gauss[i][1]))
#     print( "width = %0.7f (+/-) %0.7f" % (popt_gauss[i][2], perr_gauss[i][2]))
#     print( "area = %0.7f" % np.trapz(Gaussian(two_theta, *popt_gauss[i])))


def get_index_in_interval(datax,x_interval):
    """Get index of x interval.

    Args:
        datax (1-D array)
        x_interval (1-D list)

    Returns:
        index: index of x interval
    """
    return np.where((datax>=x_interval[0]) & (datax<x_interval[1]))[0]



"""select data from interval"""
def interval_data(two_theta,intensity,x_interval):
    interval_index = get_index_in_interval(two_theta, x_interval)
    return two_theta[interval_index], intensity[interval_index]




# """(Gaussian) fitting method 2 (fitting display well for normal gaussian distribution"""
# def fitting_method_2(two_theta,intensity,x_interval):
#     interval_index = get_index_in_interval(two_theta, x_interval)
#     x_interval_value = two_theta[interval_index]
#     y_interval_value = intensity[interval_index]
#     y_base = y_interval_value - min(y_interval_value)

#     model = GaussianModel()
#     # model = ExponentialModel()
#     # model = LorentzianModel()
#     # model = PseudoVoigtModel()
#     # model = ExponentialModel()
#     pars=model.guess(y_interval_value,x=x_interval_value)
#     # pars = model.make_params()
#     output = model.fit(y_base, pars, x=x_interval_value)
#     plt.plot(x_interval_value, y_interval_value, '-', label='original data')
#     plt.plot(x_interval_value, y_base, label='data staring at 0')
#     plt.plot(x_interval_value, output.best_fit, '--', label='fitting')
#     plt.title('Gaussian fitting for dataset %d' %i)
#     # plt.title('Lorentzian fitting for dataset %d' %i)
#     # plt.title('Voigt fitting for dataset %d' %i)
#     plt.xlabel(r'$2\theta$')
#     plt.ylabel("intensity")
#     plt.legend()
#     plt.show()




def baseline_als(y, lam, p, niter=10):
    """Define a baseline to make the distribution flat on.

    Args:
        y (_type_): Matrix with spectra in rows
        lam (_type_): 2nd derivative constraint (smoothness parameter)
        p (_type_): Weighting of positive residuals
        niter (int, optional): Maximum number of iterations. Defaults to 10.

    Returns:
        _type_: _description_
    """
    L = len(y)
    matrix = sparse.csc_matrix(np.diff(np.eye(L), 2))  #Sparse matrix in CSC format
    w = np.ones(L)  #set all numbers in array to 1
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)   #sparse.spdiags(Matrix diagonals are stored in rows, (k=0 main diagonal, k>0 The kth upper diagonal, k<0 The kth lower diagonal), result shape, result shape)
        Z = W + lam * matrix.dot(matrix.transpose())
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z




# def gaussian_fitting_curve(two_theta,intensity,x_interval,gauss1,gauss2):
def gaussian_fitting_curve(two_theta,intensity,x_interval,set_pars):
    x_interval_value, y_interval_value = interval_data(two_theta,intensity,x_interval)
    mod = GaussianModel(prefix='g1_')
    pars = mod.guess(y_interval_value, x=x_interval_value)
    pars['g1_center'].set(value=set_pars[0][0])
    pars['g1_sigma'].set(value=set_pars[0][1])
    pars['g1_amplitude'].set(value=set_pars[0][2])
    for i in range(1,len(set_pars)):
        mod_gauss = GaussianModel(prefix='g%d_' % (i+1))
        pars.update(mod_gauss.make_params())
        mod = mod+mod_gauss
        pars['g%d_center'%(i+1)].set(value=set_pars[i][0])
        pars['g%d_sigma'%(i+1)].set(value=set_pars[i][1])
        pars['g%d_amplitude'%(i+1)].set(value=set_pars[i][2])
        # mod = mod + GaussianModel(prefix='g%d_' % (i+1))
        fitting = mod.fit(y_interval_value, pars, x=x_interval_value)
    return fitting.best_fit, fitting.params.items()




def gaussian_fitting_plot(two_theta,intensity,x_interval,set_pars):
    x_interval_value, y_interval_value = interval_data(two_theta,intensity,x_interval)
    plt.plot(x_interval_value, y_interval_value, '-', label='original data')
    # plt.title('Gaussian fitting for dataset %d' %i)
    baseline = baseline_als(y_interval_value,10000,0.0001)
    baseline_subtracted = y_interval_value - baseline
    plt.plot(x_interval_value, baseline,':',label='baseline')
    plt.plot(x_interval_value, baseline_subtracted,label='after background subtraction')
    fitting,_ = gaussian_fitting_curve(x_interval_value,baseline_subtracted,x_interval,set_pars)
    plt.plot(x_interval_value, fitting, '--', label='fitting')
    plt.xlim(0,70)
    plt.ylim(0,3700)
    plt.legend()
    # plt.savefig(f"{name}_plot.png")
    plt.show()




def gaussian_plot_error(two_theta,intensity,x_interval,set_pars):
    x_interval_value, y_interval_value = interval_data(two_theta,intensity,x_interval)
    plt.plot(x_interval_value, y_interval_value, '-', label='original data')
    plt.title('Gaussian fitting result' )
    baseline = baseline_als(y_interval_value,10000,0.01)
    baseline_subtracted = y_interval_value - baseline
    # plt.plot(x_interval_value, baseline,':',label='baseline')
    # plt.plot(x_interval_value, baseline_subtracted,label='after background subtraction')
    fitting,_ = gaussian_fitting_curve(x_interval_value,baseline_subtracted,x_interval,set_pars)
    plt.plot(x_interval_value, fitting + baseline, '--', label='fitting data')
    # plt.plot(x_interval_value, fitting, '--', label='fitting')
    error = abs(baseline_subtracted - fitting)
    plt.plot(x_interval_value,error,':', label='error')
    plt.xlim(0,70)
    plt.ylim(0,3700)
    plt.legend()
    plt.savefig('Gaussian fitting result')
    plt.show()





# """gaussian fitting method 3 (used in skewing distribution)"""
#[1,3.5] interval use background subtraction


# def fitting_method_3(two_theta,intensity,x_interval):
#     interval_index = get_index_in_interval(two_theta, x_interval)
#     x_interval_value = two_theta[interval_index]
#     y_interval_value = intensity[interval_index]

#     model = GaussianModel()
#     # model = ExponentialModel()
#     # model = LorentzianModel()
#     # model = PseudoVoigtModel()
#     pars=model.guess(y_interval_value, x=x_interval_value)
#     # pars = model.make_params()
#     plt.plot(x_interval_value, y_interval_value, '-', label='original data')
#     # plt.plot(x, y_base, label='data staring at 0')
#     # plt.plot(x, output.best_fit, '--', label='fit')
#     plt.title('Gaussian fitting for dataset %d' %i)
#     # plt.title('Lorentzian fitting for dataset %d' %i)
#     # plt.title('Voigt fitting for dataset %d' %i)

#     baseline = baseline_als(y_interval_value,100000,0.01)
#     baseline_subtracted = y_interval_value - baseline
#     plt.plot(x_interval_value, baseline,':',label='baseline')
#     plt.plot(x_interval_value, baseline_subtracted,label='after background subtraction')
#     # pars1=model.guess(baseline_subtracted, x=x_interval_value)
#     fitting = model.fit(baseline_subtracted, pars, x=x_interval_value)
#     plt.plot(x_interval_value, fitting.best_fit, '--', label='fitting')
#     plt.xlim(0,5)
#     plt.ylim(0,0.1)
#     plt.legend()
#     plt.show()
#     for name, pars in fitting.params.items():
#         print(" %s: %s +/- %s " %(name,pars.value,pars.stderr))



# def fitting_method_4(two_theta,intensity,x_interval):
#     interval_index = get_index_in_interval(two_theta, x_interval)
#     x_interval_value = two_theta[interval_index]
#     y_interval_value = intensity[interval_index]
#     model = GaussianModel()
#     # model = ExponentialModel()
#     # model = LorentzianModel()
#     # model = PseudoVoigtModel()
#     pars=model.guess(y_interval_value, x=x_interval_value)
#     # pars = model.make_params()
#     baseline = baseline_als(y_interval_value,10000,0.01)
#     baseline_subtracted = y_interval_value - baseline
#     fitting = model.fit(baseline_subtracted, pars, x=x_interval_value)
#     return fitting


# """print all pars value[1,5]"""
# def print_pars_value(two_theta,intensity,x_interval):
#     fitting = fitting_method_4(two_theta,intensity,x_interval)
#     pars_value = []
#     for name, pars in fitting.params.items():
#         # print(" %s: %s +/- %s " %(name,pars.value,pars.stderr))
#         pars = pars.value
#         pars_value.append(pars)
#     return pars_value

# for i in range(intensity.shape[0]):
#     print_pars_value(two_theta,intensity[i],[6,7])


# """print all pars value[5*16]"""
# """[16*5]-[1,5]-[5,1]-[5*16]"""
# """store the pars in [5*16]list format"""
# def print_pars_value(two_theta,intensity,x_interval):
#     fitting = fitting_method_4(two_theta,intensity,x_interval)
#     pars_value = []
#     for name, pars in fitting.params.items():
#         # print(" %s: %s +/- %s " %(name,pars.value,pars.stderr))
#         pars = pars.value
#         pars_value.append(pars)
#     return pars_value

# def print_all_pars_values(two_theta,intensity,x_interval):
#     all_data = []
#     for i in range(intensity.shape[0]):
#         data = print_pars_value(two_theta,intensity[i],x_interval)
#         all_data.append(data)
#     return all_data

# all_pars_values = print_all_pars_values(two_theta,intensity,[1.2,2.5])
# all_pars_values_transposed = np.array(all_pars_values)
# all_pars_values_transposed = all_pars_values_transposed.astype(np.float32)
# all_pars_values_transposed = all_pars_values_transposed.transpose()
# print(all_pars_values_transposed)




# """print all error value[1*5]"""
# def print_error_value(two_theta,intensity,x_interval):
#     fitting = fitting_method_4(two_theta,intensity,x_interval)
#     error_value = []
#     for name, pars in fitting.params.items():
#         # print(" %s: %s +/- %s " %(name,pars.value,pars.stderr))
#         error = pars.stderr
#         error_value.append(error)
#     return error_value

# for i in range(intensity.shape[0]):
#     print_error_value(two_theta,intensity[i],[1.2,2.5])



# """print all error value[5*16]"""
# """[16*5]-[1,5]-[5,1]-[5*16]"""
# """store the error in [5*16]list format"""
# def print_error_value(two_theta,intensity,x_interval):
#     fitting = fitting_method_4(two_theta,intensity,x_interval)
#     error_value = []
#     for name, pars in fitting.params.items():
#         # print(" %s: %s +/- %s " %(name,pars.value,pars.stderr))
#         error = pars.stderr
#         error_value.append(error)
#     return error_value

#     # return 0 if not_float(error_value) else error_value
#     # 0 if value is None else value

# def print_all_error_values(two_theta,intensity,x_interval):
#     all_data = []
#     for i in range(intensity.shape[0]):
#         data = print_error_value(two_theta,intensity[i],x_interval)
#         all_data.append(data)
#     return all_data

# all_error_values = print_all_error_values(two_theta,intensity,[1.2,2.5])
# all_error_values_transposed = np.array(all_error_values)
# all_error_values_transposed = all_error_values_transposed.astype(np.float32).transpose()
# all_error_values_transposed =  np.where(np.isnan(all_error_values_transposed), 0, all_error_values_transposed)
# print(all_error_values_transposed)





# """store the pars and error in [10*16]list format"""
# """easy to export to csv file"""
# def print_par_error(two_theta,intensity,x_interval):
#     fitting = fitting_method_4(two_theta,intensity,x_interval)
    
#     par_err = []
#     for name, pars in fitting.params.items():
#         # print(" %s: %s +/- %s " %(name,pars.value,pars.stderr))
#         error = pars.stderr
#         pars = pars.value
#         par_err.append(pars)
#         par_err.append(error)
#     return par_err

# def print_all_par_error(two_theta,intensity,x_interval):
#     all_data = []
#     for i in range(intensity.shape[0]):
#         data = print_par_error(two_theta,intensity[i],x_interval)
#         all_data.append(data)
#     return all_data

# all_par_error = print_all_par_error(two_theta,intensity,[1.2,2.5])
# all_par_error_transposed = np.array(all_par_error)
# all_par_error_transposed = all_par_error_transposed.astype(np.float32).transpose()
# all_par_error_transposed =  np.where(np.isnan(all_par_error_transposed), 0, all_par_error_transposed)
# print(all_par_error_transposed)


# """export amplitude,center,sigma,fwhm, height and error into csv file"""
# rows = all_par_error_transposed
# np.savetxt("par_error.csv", 
#            rows,
#            delimiter =", ", 
#            fmt ='% s')


#     #print(fitting.fit_report())
#     """return amplitude(represents the overall intensity (or area of) a peak or function)
#        return sigma parameter that gives a characteristic width."""
#     for name, pars in fitting.params.items():
#         print(" %s: value=%s +/- %s " % (name, pars.value, pars.stderr))
    
#     #only print one params
#     amplitude_value = fitting.params['amplitude'].value
#     center_value = fitting.params['center'].value
#     sigma_value = fitting.params['sigma'].value
#     fwhm_value = fitting.params['fwhm'].value
#     height_value = fitting.params['height'].value

#     amplitude.append(amplitude_value)
#     center.append(center_value)
#     sigma.append(sigma_value)
#     fwhm.append(fwhm_value)
#     height.append(height_value)

# """export amplitude,center,sigma,fwhm and height into csv file"""
# def all_to_csv_file(col1,col2,col3,col4,col5,col6,name):
#     data_file = pd.DataFrame({'time':col1,'amplitude':col2,'center':col3,'sigma':col4,'fwhm':col5,'height':col6})
#     data_file.to_csv(name+'.csv',index=0,sep=',')

# time = list(range(0,170,10))
# all_to_csv_file(time,amplitude,center,sigma,fwhm,height,'all params')


# """export only one parameter(like fwhm) to csv file and plot the variation in it"""
# def to_csv_file(col1,col2,name):
#     data_file = pd.DataFrame({'time':col1,name:col2})
#     data_file.to_csv(name+'.csv',index=0,sep=',',header=None)



"""plot variation in fwhm"""
# time = list(range(0,170,10))
# fwhm = all_pars_values_transposed[3]
time = list(range(30,170,10))
fwhm = all_pars_values_transposed[3][3:] #ignore the first three datasets because there are no peaks.
# to_csv_file(time,fwhm,'fwhm')
plt.plot(time, fwhm)
plt.scatter(time, fwhm)
plt.errorbar(time, fwhm, yerr=all_error_values_transposed[3][3:])
plt.title("variation in fwhm")
plt.xlabel("time")
plt.ylabel("fwhm")
plt.show()



"""plot variation in height"""
# time = list(range(0,170,10))
# height = all_pars_values_transposed[4]
time = list(range(30,170,10))
height = all_pars_values_transposed[4][3:]
plt.plot(time, height)
plt.scatter(time, height)
plt.errorbar(time, height, yerr=all_error_values_transposed[4][3:])
plt.title("variation in height")
plt.xlabel("time")
plt.ylabel("height")
plt.show()
