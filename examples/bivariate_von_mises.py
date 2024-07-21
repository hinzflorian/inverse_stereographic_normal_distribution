"""Fitting a bivariate von Mises distribution to data.
"""

import json
from torch.utils.data import DataLoader
import torch
import pyro.distributions as dist
from isnd.visualization.visualize_distribution import heat_plot
from isnd.visualization.visualize_torus import plot_heatmap_on_torus
import math
import numpy as np
import datetime
import os
from isnd.training.train_prior import train
from isnd.loss.loss import ml_loss
import time
from isnd.model.von_mises.independent_vonMises import IndependentVonMisesMixtureModel
from isnd.model.isnd import IsndMixtureModel
from isnd.clustering.clustering import cluster_on_torus
from torch.utils.data import DataLoader


if __name__ == "__main__":
    device = 5
    draw_samples = False
    if draw_samples:
        phi_loc = torch.tensor(3.0)  # Location parameter for phi
        psi_loc = torch.tensor(3.0)  # Location parameter for psi
        concentration1 = torch.tensor(0.8)  # Concentration parameter for phi
        concentration0 = torch.tensor(1.5)  # Concentration parameter for psi
        correlation = torch.tensor(2.5)  # Correlation between phi and psi
        distribution = dist.SineBivariateVonMises(
            phi_loc,
            psi_loc,
            concentration0,
            concentration1,
            correlation,
        )
        num_samples = 1000000
        samples = distribution.sample((num_samples,))
        save_dir="datasets/sine_bivariate_von_mises/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir,"sine_bivariate_von_mises_samples.npy")
        torch.save(samples, save_path)
    ###############################################################################################
    # fit the distribution
    ###############################################################################################
    # load data
    data_path = "datasets/sine_bivariate_von_mises/sine_bivariate_von_mises_samples.npy"
    data = torch.load(data_path)
    # device = "cpu"
    network_paramterization = True
    n_dim = data.shape[1]
    # split data into train and test set
    total_indices = torch.randperm(data.shape[0])
    train_size = int(0.8 * data.shape[0])
    train_indices = total_indices[:train_size]
    test_indices = total_indices[train_size:]
    samples_train = data[train_indices]
    samples_test = data[test_indices]
    #################################################################
    train_loader = DataLoader(
        samples_train, shuffle=True, batch_size=40000, num_workers=0
    )
    #################################################################
    # fit data with isnd
    #################################################################
    data_type = "sine_bivariate_von_mises"
    n_dim = data.shape[1]
    n_components = 5
    num_epochs = 50
    # model_type = "von_mises"
    model_type = "isnd"
    training = True
    if training:
        cluster_centers = cluster_on_torus(data, n_components)
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
        ###########################################################################################
        # save model
        ###########################################################################################
        if not os.path.exists(checkpoint_directory):
            os.makedirs(checkpoint_directory)
        torch.save(
            model.state_dict(),
            checkpoint_directory
            + f"{model_type}_{n_components}_{num_epochs}_epochs_{n_dim}_dim.pth",
        )
    ###############################################################################################
    # save model
    ###############################################################################################
    plotting = False
    if plotting:
        data_path = "images/sine_bivariate_von_mises/sine_bivariate_von_mises_samples.npy"
        samples = torch.load(data_path)
        samples_shifted = samples % (2 * math.pi)
        directory_name = "images/"
        filename = "sin_bivariate_von_mises"
        save_path = directory_name + filename + "_on_torus.png"
        heat_plot(
            samples_shifted[:, 0], samples_shifted[:, 1], directory_name, filename
        )
        #################################################################
        # load model
        #################################################################
        model_paths = "isnd/config/checkpoint_bivariate_von_mises.json"
        with open(model_paths, "r") as file:
            checkpoint_paths = json.load(file)
        components_list = [100, 150, 200]
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
        # draw samples
        n_samples = data.shape[0]
        samples_model = (
            model.sample_from_density(n_samples)
            .to(device, dtype=torch.float32)
            .detach()
        )
        directory_name = "images/"
        filename = f"sin_bivariate_von_mises_fitted_{n_components}_components.png"
        heat_plot(
            samples_model[:, 0].cpu(),
            samples_model[:, 1].cpu(),
            directory_name,
            filename,
        )
