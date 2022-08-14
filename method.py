import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import codecs
import os

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




def read_text_file(file_path):
    with codecs.open(file_path, mode='r', encoding="utf-8-sig") as file:
        data = np.loadtxt(file, skiprows=25, dtype=float)
        return data

def open_file(data_path):
    files = os.listdir(data_path) # get all file name under the folder
    files.sort()  # read the file in order
    # print(files)
    two_theta = []
    intensities = []
    for file in files: #traverse folder
        if file[-3:] == '.gr':
        # if not os.path.isdir(file): #determine whether it is a number, if not open it
            file_path = data_path+"/"+file #open file
                # with codecs.open(file_path, mode='r', encoding="utf-8-sig",errors='ignore') as file:
                # with codecs.open(file_path, mode='r', encoding="utf-8-sig") as file:
                #     data = np.loadtxt(file, skiprows=25, dtype=float)
                #     print(data)
            # print(file_path)
            line = read_text_file(file_path).transpose()
            two_theta = line[0]
            intensities.append(line[1])

    # data_transposed = np.array(data).transpose()
    # data_transposed = data_transposed.astype(np.float32)
    # return data_transposed 
    return two_theta, intensities




def separate_x_y_axis(data):
    """To separate the data into two-theta(x-axis) and intensities(y-axis).

    Args:
        data (2-D array): the first line is two_theta(x-axis) and others are intensity(y-axis).

    Returns:
        two_theta, intensity: two theta(x-axis) and intensities in different temperature(y-axis).
    """
    two_theta = data[0]
    intensity = data[1]
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
    for i in range(len(intensity)):
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
    for i in range(len(intensity)):
        plt.loglog(two_theta, intensity[i], linewidth = 0.5)
        # plt.plot(two_theta_log, intensity_log[i], linewidth = 0.5)
    plt.title("intensity")
    plt.xlabel(r'$2\theta$')
    plt.ylabel("intensity")
    plt.show()


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
    # baseline = baseline_als(y_interval_value,10000,0.0001)
    baseline = baseline_als(y_interval_value,10000,0.01)
    baseline_subtracted = y_interval_value - baseline
    plt.plot(x_interval_value, baseline,':',label='baseline')
    plt.plot(x_interval_value, baseline_subtracted,label='after background subtraction')
    fitting,_ = gaussian_fitting_curve(x_interval_value,baseline_subtracted,x_interval,set_pars)
    plt.plot(x_interval_value, fitting, '--', label='fitting')
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
    plt.legend()
    plt.savefig('./fitting result/Gaussian fitting result')
    plt.show()



#lorentzian
def lorentzian_fitting_curve(two_theta,intensity,x_interval,set_pars):
    x_interval_value, y_interval_value = interval_data(two_theta,intensity,x_interval)
    mod = LorentzianModel(prefix='l1_')
    pars = mod.guess(y_interval_value, x=x_interval_value)
    pars['l1_center'].set(value=set_pars[0][0])
    pars['l1_sigma'].set(value=set_pars[0][1])
    pars['l1_amplitude'].set(value=set_pars[0][2])
    for i in range(1,len(set_pars)):
        mod_gauss = LorentzianModel(prefix='l%d_' % (i+1))
        pars.update(mod_gauss.make_params())
        mod = mod+mod_gauss
        pars['l%d_center'%(i+1)].set(value=set_pars[i][0])
        pars['l%d_sigma'%(i+1)].set(value=set_pars[i][1])
        pars['l%d_amplitude'%(i+1)].set(value=set_pars[i][2])

        fitting = mod.fit(y_interval_value, pars, x=x_interval_value)
    return fitting.best_fit, fitting.params.items()




def lorentzian_fitting_plot(two_theta,intensity,x_interval,set_pars):
    x_interval_value, y_interval_value = interval_data(two_theta,intensity,x_interval)
    plt.plot(x_interval_value, y_interval_value, '-', label='original data')
    # plt.title('Lorentzian fitting for dataset %d' %i)
    baseline = baseline_als(y_interval_value,10000,0.01)
    baseline_subtracted = y_interval_value - baseline
    plt.plot(x_interval_value, baseline,':',label='baseline')
    plt.plot(x_interval_value, baseline_subtracted,label='after background subtraction')
    fitting,_ = lorentzian_fitting_curve(x_interval_value,baseline_subtracted,x_interval,set_pars)
    plt.plot(x_interval_value, fitting, '--', label='fitting')
    plt.legend()
    # plt.savefig(f"{name}_plot.png")
    plt.show()


def lorentzian_plot_error(two_theta,intensity,x_interval,set_pars):
    x_interval_value, y_interval_value = interval_data(two_theta,intensity,x_interval)
    plt.plot(x_interval_value, y_interval_value, '-', label='original data')
    plt.title('Lorentzian fitting result' )
    baseline = baseline_als(y_interval_value,10000,0.01)
    baseline_subtracted = y_interval_value - baseline
    # plt.plot(x_interval_value, baseline,':',label='baseline')
    # plt.plot(x_interval_value, baseline_subtracted,label='after background subtraction')
    fitting,_ = lorentzian_fitting_curve(x_interval_value,baseline_subtracted,x_interval,set_pars)
    plt.plot(x_interval_value, fitting + baseline, '--', label='fitting data')
    # plt.plot(x_interval_value, fitting, '--', label='fitting')
    error = abs(baseline_subtracted - fitting)
    plt.plot(x_interval_value,error,':', label='error')
    plt.legend()
    plt.savefig('./fitting result/Lorentzian fitting result')
    plt.show()



#PseudoVoigt
def PseudoVoigt_fitting_curve(two_theta,intensity,x_interval,set_pars):
    x_interval_value, y_interval_value = interval_data(two_theta,intensity,x_interval)
    mod = PseudoVoigtModel(prefix='p1_')
    pars = mod.guess(y_interval_value, x=x_interval_value)
    pars['p1_center'].set(value=set_pars[0][0])
    pars['p1_sigma'].set(value=set_pars[0][1])
    pars['p1_amplitude'].set(value=set_pars[0][2])
    for i in range(1,len(set_pars)):
        mod_gauss = PseudoVoigtModel(prefix='p%d_' % (i+1))
        pars.update(mod_gauss.make_params())
        mod = mod+mod_gauss
        pars['p%d_center'%(i+1)].set(value=set_pars[i][0])
        pars['p%d_sigma'%(i+1)].set(value=set_pars[i][1])
        pars['p%d_amplitude'%(i+1)].set(value=set_pars[i][2])

        fitting = mod.fit(y_interval_value, pars, x=x_interval_value)
    return fitting.best_fit, fitting.params.items()


def PseudoVoigt_fitting_plot(two_theta,intensity,x_interval,set_pars):
    x_interval_value, y_interval_value = interval_data(two_theta,intensity,x_interval)
    plt.plot(x_interval_value, y_interval_value, '-', label='original data')
    # plt.title('PseudoVoigt fitting for dataset %d' %i)
    baseline = baseline_als(y_interval_value,10000,0.01)
    baseline_subtracted = y_interval_value - baseline
    plt.plot(x_interval_value, baseline,':',label='baseline')
    plt.plot(x_interval_value, baseline_subtracted,label='after background subtraction')
    fitting,_ = PseudoVoigt_fitting_curve(x_interval_value,baseline_subtracted,x_interval,set_pars)
    plt.plot(x_interval_value, fitting, '--', label='fitting')
    plt.legend()
    # plt.savefig(f"{name}_plot.png")
    plt.show()


def PseudoVoigt_plot_error(two_theta,intensity,x_interval,set_pars):
    x_interval_value, y_interval_value = interval_data(two_theta,intensity,x_interval)
    plt.plot(x_interval_value, y_interval_value, '-', label='original data')
    plt.title('PseudoVoigt fitting result' )
    baseline = baseline_als(y_interval_value,10000,0.01)
    baseline_subtracted = y_interval_value - baseline
    # plt.plot(x_interval_value, baseline,':',label='baseline')
    # plt.plot(x_interval_value, baseline_subtracted,label='after background subtraction')
    fitting,_ = PseudoVoigt_fitting_curve(x_interval_value,baseline_subtracted,x_interval,set_pars)
    plt.plot(x_interval_value, fitting + baseline, '--', label='fitting data')
    # plt.plot(x_interval_value, fitting, '--', label='fitting')
    error = abs(baseline_subtracted - fitting)
    plt.plot(x_interval_value,error,':', label='error')
    plt.legend()
    plt.savefig('./fitting result/PseudoVoigt fitting result')
    plt.show()


def gaussian_fitting_value(two_theta,intensity,x_interval,set_pars):
    dic = {}
    x_interval_value, y_interval_value = interval_data(two_theta,intensity,x_interval)
    baseline = baseline_als(y_interval_value,10000,0.01)
    baseline_subtracted = y_interval_value - baseline
    _,fitting_params = gaussian_fitting_curve(x_interval_value,baseline_subtracted,x_interval,set_pars)
    for name, pars in fitting_params:
        if pars.value is not None:
            pars.value = pars.value
        else:
            pars.value = 0

        if pars.stderr is not None:
            pars.stderr = pars.stderr
        else:
            pars.stderr = 0
        # pars.value = np.where(np.isnan(pars.value), 0, pars.value)
        # pars.stderr = np.where(np.isnan(pars.stderr), 0, pars.stderr)
        key1 = name
        key2 = name+'error'
        if key1 not in dic:
            dic[key1] = []
        if key2 not in dic:
            dic[key2] = []
        dic[key1].append(pars.value)
        dic[key2].append(pars.stderr)
        # print (" %s: %0.6f +/- %0.6f " %(name,pars.value,pars.stderr))
    return dic




def mergeDic(dicT,dic):
    for key in dic:
        if key not in dicT:
            dicT[key] = dic[key]
        dicT[key] = dicT[key] + dic[key]
    return dicT




def getCsv(dicT,i):
    datas = pd.DataFrame(dicT)
    peaks = {}
    for i in range(len(datas.columns)//10):
        # exec(f'peak{i+1}' = datas[datas.columns[10*i:10*(i+1)]])
        peaks['peak{}'.format(i+1)] = datas[datas.columns[10*i:10*(i+1)]]
        # 'peaktry%d'%i = datas[datas.columns[0:10*(i+1)]]
        # 'peak{}'.format(i+1).to_csv("'peak{}'.format(i+1).csv",index=False)
        # "peak{}.csv".format(i+1)
        # peaks['peak{}'.format(i+1)].to_csv("'peak{}'.format(i+1).csv",index=False)
        peaks['peak{}'.format(i+1)].to_csv("./peakFiles/peak{}.csv".format(i+1),index=False)
    return peaks

# peak1 = getCsv(dicT,3)




def toCsv(two_theta,intensity,x_interval,set_pars):
    dicT = {}
    for i in range(len(intensity)):
        tDic = gaussian_fitting_value(two_theta,intensity[i],x_interval,set_pars)
        mergeDic(dicT,tDic)
    getCsv(dicT,len(set_pars))



    

def chisquare(obs, exp):
    """fitting index using chi square method.

    Args:
        obs ([array]): observed value 
        exp ([type]): expected value

    Returns:
        [array]: fitting index
    """    """"""
    obs = np.atleast_1d(np.asanyarray(obs))
    exp = np.atleast_1d(np.asanyarray(exp)) # convert list to array
    if obs.size != exp.size:
        print('The size of the observed array and the expected array is not equal')
        exit()
    return ((obs - exp) ** 2 / exp).sum(axis=0)




def gaussian_fit_index(two_theta,intensity,x_interval,set_pars):
    x_interval_value, y_interval_value = interval_data(two_theta,intensity,x_interval)
    baseline = baseline_als(y_interval_value,10000,0.01)
    baseline_subtracted = y_interval_value - baseline
    fitting,_ = gaussian_fitting_curve(x_interval_value,baseline_subtracted,x_interval,set_pars)

    observed = y_interval_value
    expected = fitting + baseline

    return chisquare(observed,expected)



#plot change in fwhm
def change_fwhm(csvFile):
    tPath = os.path.join('peakFiles',csvFile)
    csvPre = csvFile.split('.')[0]
    readData = pd.read_csv(tPath)   
    fwhm = readData.iloc[:,6].tolist()  
    # time = list(range(0,220,10))  
    time = list(range(0,len(fwhm)*10,10))        
    error = readData.iloc[:,7].tolist()
    plt.plot(time, fwhm,'r-^')            
    plt.errorbar(time, fwhm, yerr=error)
    plt.title("change in FWHM in {}".format(csvPre))              
    plt.xlabel("time")                
    plt.ylabel("FWHM") 
    plt.savefig("./change in FWHM and height/change in FWHM in {}".format(csvPre))                 
    plt.show()

def all_change_fwhm():
    Files = []
    for root,dirs,files in os.walk('./peakFiles'):
        for name in files:
            Files.append(name)
    for f in Files:
        change_fwhm(f)

#plot change in height
def change_height(csvFile):
    tPath = os.path.join('peakFiles',csvFile)
    csvPre = csvFile.split('.')[0]
    readData = pd.read_csv(tPath)
    height = readData.iloc[:,8].tolist()  
    # time = list(range(0,220,10)) 
    time = list(range(0,len(height)*10,10))         
    error = readData.iloc[:,9].tolist()
    plt.plot(time, height,'r-^')            
    plt.errorbar(time, height, yerr=error)
    plt.title("change in height in {}".format(csvPre))              
    plt.xlabel("time")                
    plt.ylabel("height") 
    plt.savefig("./change in FWHM and height/change in height in {}".format(csvPre))                 
    plt.show() 



def all_change_height():
    Files = []
    for root,dirs,files in os.walk('./peakFiles'):
        for name in files:
            Files.append(name)
    for f in Files:
        change_height(f)

