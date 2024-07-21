"""Trivariate wind data. Obtained from the R package "CircNNTSR" via the comman("WindDirectionsTrivariate", package = "CircNNTSR")
"""

import torch
import numpy as np
from isnd.visualization.visualize_distribution import heat_plot, plot_marginals

from isnd.training.train_prior import train
from isnd.loss.loss import ml_loss
from isnd.model.isnd import IsndMixtureModel
from isnd.model.von_mises.independent_vonMises import IndependentVonMisesMixtureModel
import time
from isnd.clustering.clustering import cluster_on_torus
import os
from torch.utils.data import DataLoader

if __name__ == "__main__":
    # draw samples from normal distribution
    device = 0
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    data_path = (
        "datasets/wind/WindDirectionsTrivariate.csv"
    )
    wind_data = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    #############################################################################
    # train isnd model
    #############################################################################
    wind_data_tensor = torch.tensor(wind_data).to(device)
    train_loader = DataLoader(
        wind_data_tensor, shuffle=True, batch_size=40000, num_workers=0
    )
    ###############################
    # fit model
    ###############################
    data_type = "wind"
    n_dim = wind_data.shape[1]
    n_components = 50
    num_epochs = 500
    model_type = "isnd"
    training = True
    if training == True:
        cluster_centers = cluster_on_torus(wind_data_tensor, n_components)
        # model_type = "von_mises"
        if model_type == "isnd":
            model = IsndMixtureModel(
                n_dim, n_components, cluster_centers, orth_parametrization="matrix_exp"
            )
        elif model_type == "von_mises":
            model = IndependentVonMisesMixtureModel(
                n_dim, n_components, cluster_centers, initial_weights=None
            )
        model.to(device)
        model = model.double()
        learning_rate = 0.01
        loss_function = ml_loss
        # train the model:
        start_time = time.time()
        train(
            model,
            learning_rate,
            train_loader,
            num_epochs,
            loss_function,
        )
        end_time = time.time()
        print(
            f"Time taken for {n_components} components: {(end_time - start_time)/60} minutes"
        )
        # save model
        checkpoint_directory = f"isnd/checkpoints/{data_type}/{model_type}/"
        ###############################################################################################
        # save model
        ###############################################################################################
        # Write string to file
        if not os.path.exists(checkpoint_directory):
            os.makedirs(checkpoint_directory)
        torch.save(
            model.state_dict(),
            checkpoint_directory
            + f"{model_type}_{n_components}_{num_epochs}_epochs_{n_dim}_dim.pth",
        )
    ###########################################################################################
    # plotting
    ###########################################################################################
    plotting = False
    if plotting:
        directory = "images/wind/"
        file_name = "wind_data.png"
        plot_marginals(wind_data.T, directory, format="png")
        # plot fitted
        samples_model = (
            model.sample_from_density(wind_data.shape[0])
            .to(device, dtype=torch.float32)
            .detach()
        ).cpu()
        directory = "images/wind/fitted"
        file_name = "fitted_wind_data.png"
        plot_marginals(samples_model.T, directory, format="png")
