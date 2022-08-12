import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from method import(
    read_data,
    open_file,
    separate_x_y_axis,
    plot_data,
    plot_data_log10,
    baseline_als,
    get_index_in_interval,
    interval_data,
    gaussian_fitting_curve,
    gaussian_fitting_plot,
    gaussian_plot_error,
    lorentzian_fitting_curve,
    lorentzian_fitting_plot,
    lorentzian_plot_error,
    PseudoVoigt_fitting_curve,
    PseudoVoigt_fitting_plot,
    PseudoVoigt_plot_error,
    gaussian_fitting_value,
    mergeDic,
    getCsv,
    chisquare,
    gaussian_fit_index
)


def main():
    # filename = "ZnO transformation.csv"
    # data = read_data(filename)
    # filename = "PdCeO2"
    # data = open_file(filename)
    # two_theta, intensity = separate_x_y_axis(data)
    # plot_data(two_theta, intensity)
    # gaussian_fitting_plot(two_theta,intensity[50],[20,35],[[26.4,1,600],[28.4,1,750],[30,1,750]])
    # gaussian_plot_error(two_theta,intensity[60],[20,35],[[26.4,1,600],[28.4,1,750],[30,1,750]])




    filename = "PdCeO2"
    data = open_file(filename)
    two_theta, intensity = separate_x_y_axis(data)
    plot_data(two_theta, intensity)
    #plot every dataset in the folder
    for i in range(len(intensity)):
        plt.title('Gaussian fitting for dataset %d' %i)
        gaussian_fitting_plot(two_theta,intensity[i],[1.8,4.2],[[2.4,0.038,0.3],[3.8,0.07,1.13]])

    gaussian_plot_error(two_theta,intensity[3],[1.8,4.2],[[2.4,0.038,0.3],[3.8,0.07,1.13]])
    lorentzian_plot_error(two_theta,intensity[3],[1.8,4.2],[[2.4,0.038,0.3],[3.8,0.07,1.13]])
    PseudoVoigt_plot_error(two_theta,intensity[3],[1.8,4.2],[[2.4,0.038,0.3],[3.8,0.07,1.13]])








if __name__ == "__main__":
    main()