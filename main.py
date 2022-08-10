from method import(
    read_data,
    separate_x_y_axis,
    plot_data,
    plot_data_log10,
    baseline_als,
    get_index_in_interval,
    interval_data,
    gaussian_fitting_curve,
    gaussian_fitting_plot,
    gaussian_plot_error,
    gaussian_fitting_value,
    mergeDic,
    getCsv,
    chisquare,
    gaussian_fit_index
)


def main():
    filename = "ZnO transformation.csv"
    data = read_data(filename)
    two_theta, intensity = separate_x_y_axis(data)
    plot_data(two_theta, intensity)
    gaussian_fitting_plot(two_theta,intensity[50],[20,35],[[26.4,1,600],[28.4,1,750],[30,1,750]])
    gaussian_plot_error(two_theta,intensity[60],[20,35],[[26.4,1,600],[28.4,1,750],[30,1,750]])











if __name__ == "__main__":
    main()