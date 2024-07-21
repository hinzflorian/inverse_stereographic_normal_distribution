"""Generate plots for sine bivariate von mises distribution
"""

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
    data_path = "datasets/sine_bivariate_von_mises/sine_bivariate_von_mises_samples.npy"
    samples = torch.load(data_path)
    samples_shifted = samples % (2 * math.pi)
    permuted_indices = torch.randperm(samples_shifted.shape[0])
    samples_shifted = samples_shifted[permuted_indices]
    #################################################################
    # create plots for true samples
    #################################################################
    directory_name = (
        "images/sine_bivariate_von_mises/"
    )
    filename = "sin_bivariate_von_mises"
    save_path = directory_name + filename + "_on_torus.png"
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    heat_plot(samples_shifted[:, 0], samples_shifted[:, 1], directory_name, filename)
    title = "Target Distribution"
    file_prefix = f"true_distribution"
    bw_adjust = 1.8
    all_density_plots(
        samples_shifted[:10000], directory_name, title, file_prefix, bw_adjust
    )
    #################################################################
    # load model
    #################################################################
    model_paths = "isnd/config/checkpoint_bivariate_von_mises.json"
    with open(model_paths, "r") as file:
        checkpoint_paths = json.load(file)
    components_list = [5, 10, 20, 100, 150, 200]
    ind = 0
    n_components = components_list[ind]

    model = IsndMixtureModel(
        n_dim,
        n_components,
        network_paramterization=network_paramterization,
    )
    model.to(torch.float64)
    model.to(device)
    checkpoint_path = (
        "isnd/checkpoints/sine_bivariate_von_mises/isnd/"
        + checkpoint_paths["bivariate_von_mises"][ind]
    )
    model.load_state_dict(torch.load(checkpoint_path))
    #################################################################
    # draw samples and create heat map
    #################################################################
    n_samples = samples.shape[0]
    samples_model = (
        model.sample_from_density(n_samples).to(device, dtype=torch.float32).detach()
    )
    directory_name = "images/sine_bivariate_von_mises/"
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    filename = f"sin_bivariate_von_mises_fitted_{n_components}_components.png"
    heat_plot(
        samples_model[:, 0].cpu(), samples_model[:, 1].cpu(), directory_name, filename
    )
    title = f"Fitted distribution {n_components} components"
    file_prefix = f"{n_components}_components"
    all_density_plots(
        samples_model[:10000].cpu(), directory_name, title, file_prefix, bw_adjust
    )
