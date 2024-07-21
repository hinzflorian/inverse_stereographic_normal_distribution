"""Visualize the wind distribution fit
"""

import numpy as np
import os
import json
import torch
from isnd.visualization.visualize_distribution import heat_plot
import math
from isnd.model.isnd import IsndMixtureModel
from isnd.visualization.contour_plots import all_density_plots

if __name__ == "__main__":
    device = 0
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    network_paramterization = True
    n_dim = 2
    data_path = (
        "datasets/wind/WindDirectionsTrivariate.csv"
    )
    wind_data = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    samples_shifted = wind_data % (2 * math.pi)
    permuted_indices = torch.randperm(samples_shifted.shape[0])
    samples_shifted = samples_shifted[permuted_indices]
    #################################################################
    # create plots for true samples
    #################################################################
    directory_name = "images/wind/"
    filename = "wind_data"
    save_path = directory_name + filename + "_.png"
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    heat_plot(samples_shifted[:, 0], samples_shifted[:, 1], directory_name, filename)
    title = "Target Distribution"
    file_prefix = f"true_distribution"
    bw_adjust = 1.8
    all_density_plots(samples_shifted, directory_name, title, file_prefix, bw_adjust)