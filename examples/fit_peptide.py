"""Fit the sisnd distribution for increasing number of components on torsion angle data of a peptide.
"""

import datetime
import gc
import math
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from isnd.clustering.clustering import cluster_on_torus
from isnd.model.von_mises.independent_vonMises import IndependentVonMisesMixtureModel
from isnd.loss.loss import ml_loss
from isnd.model.isnd import IsndMixtureModel
from isnd.training.train_prior import train

if __name__ == "__main__":
    device = 0
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    # Load the data
    data_path = "datasets/alaninetetrapeptide/alanine_tetrapeptide.npy"
    data = torch.tensor(np.load(data_path)) % (2 * math.pi)
    data = data[torch.randperm(data.shape[0]), :].to(device)
    data_type = data_path.split("/")[1]
    #################################################################
    # Prepare data set
    #################################################################
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
    num_epochs = 5
    min_components = 10
    max_components = 20
    # max_components = 310
    for n_components in range(min_components, max_components, 10):
        cluster_centers = cluster_on_torus(samples_train[:300000], n_components)
        model_type = "isnd"
        # model_type = "von_mises"
        if model_type == "isnd":
            model = IsndMixtureModel(
                n_dim,
                n_components,
                cluster_centers,
                network_paramterization=network_paramterization,
                orth_parametrization="matrix_exp",
            )
        elif model_type == "von_mises":
            model = IndependentVonMisesMixtureModel(
                n_dim, n_components, cluster_centers, initial_weights=None
            )
        model.to(device)
        model = model.double()
        learning_rate = 0.01
        loss_function = ml_loss
        ###########################################################################################
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
        ###########################################################################################
        # plot the model:
        # n_samples_train = 100000
        n_samples_train = samples_train.shape[0]
        # checkpoint_directory = f"isnd/checkpoints/{data_type}/"
        checkpoint_directory = f"isnd/checkpoints/{data_type}/{model_type}/"
        ###########################################################################################
        # save model
        ###########################################################################################
        # Write string to file
        if not os.path.exists(checkpoint_directory):
            os.makedirs(checkpoint_directory)
        now = datetime.datetime.now()
        now_str = now.strftime("%Y%m%d_%H%M%S")  # format: YYYYMMDD_HHMMSS
        torch.save(
            model.state_dict(),
            checkpoint_directory
            + f"{model_type}_{n_components}_{num_epochs}_epochs_{now_str}_small_eigenvalue_network_param_{network_paramterization}_learning_rate_{learning_rate}.pth",
        )
        del model  # delete the model
        torch.cuda.empty_cache()  # clear unused GPU memory
        gc.collect()
