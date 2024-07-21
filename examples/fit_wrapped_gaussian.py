"""In this experiment we want to fit a wrapped normal 
"""

import time
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from isnd.visualization.visualize_torus import plot_heatmap_on_torus
from isnd.model.von_mises.independent_vonMises import IndependentVonMisesMixtureModel
import math
from isnd.loss.loss import ml_loss
from torch.utils.data import DataLoader
import os
from isnd.clustering.clustering import cluster_on_torus
from isnd.model.isnd import IsndMixtureModel
from isnd.visualization.visualize_distribution import heat_plot
from isnd.training.train_prior import train


def create_random_cov_matrix(eigenvalues):
    # Step 1: Create a diagonal matrix of eigenvalues
    D = torch.diag(torch.tensor(eigenvalues))
    # Step 2: Create a random matrix and orthogonalize it to get Q
    A = torch.randn(D.size(0), D.size(0))
    Q, _ = torch.linalg.qr(A)
    # Step 3: Compute the covariance matrix as Q D Q^T
    cov_matrix = Q @ D @ Q.T
    return cov_matrix


if __name__ == "__main__":
    # draw samples from normal distribution
    device = 0
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    sampling = True
    data_dir = (
        "datasets/wrapped_normal"
    )
    data_path= os.path.join(data_dir,"wrapped_normal_samples.pth")
    if sampling:
        #eigenvalues = [0.1, 0.2, 0.3, 0.5, 0.7]
        eigenvalues = [0.1, 0.2]
        cov_matrix = 0.5 * create_random_cov_matrix(eigenvalues)
        mean = torch.ones(len(eigenvalues))
        # Define the multivariate normal distribution
        multivariate_gaussian = MultivariateNormal(mean, covariance_matrix=cov_matrix)
        num_samples = 1000000  # Number of samples to draw
        samples = multivariate_gaussian.sample((num_samples,))
        samples_wrapped = samples % (2 * math.pi)
        samples_wrapped = samples_wrapped.to(device)
        if not os.path.exists(data_dir):
                os.makedirs(data_dir)
        torch.save(samples_wrapped, data_path)
    samples_wrapped = torch.load(data_path)
    train_loader = DataLoader(
        samples_wrapped, shuffle=True, batch_size=40000, num_workers=0
    )
    ###############################
    # fit model
    ###############################
    data_type = "wrapped_normal"
    n_dim = len(eigenvalues)
    n_components = 1
    num_epochs = 50
    # model_type = "von_mises"
    model_type = "isnd"
    training = True
    if training == True:
        cluster_centers = cluster_on_torus(samples_wrapped[:300000], n_components)
        # model_type = "von_mises"
        if model_type == "isnd":
            model = IsndMixtureModel(
                n_dim,
                n_components,
                cluster_centers,
                network_paramterization=True,
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
        ###############################################################################################
        # Write string to file
        if not os.path.exists(checkpoint_directory):
            os.makedirs(checkpoint_directory)
        torch.save(
            model.state_dict(),
            checkpoint_directory
            + f"{model_type}_{n_components}_components_{num_epochs}_epochs.pth",
        )
    ###############################
    # create plots
    ###############################
    plotting = False
    if plotting:
        # plot first true samples
        directory = "images/fit_wrapped_gaussian/"
        if not os.path.exists(checkpoint_directory):
            os.makedirs(checkpoint_directory)
        file_name = "wrapped_normal_samples.png"
        heat_plot(
            samples_wrapped[:, 0].cpu(),
            samples_wrapped[:, 1].cpu(),
            directory,
            file_name,
            lower_limit=None,
        )
        # plot model
        # draw samples
        samples_model = (
            model.sample_from_density(num_samples)
            .to(device, dtype=torch.float32)
            .detach()
        ).cpu()
        file_name = f"fitted_{model_type}_{n_components}.png"
        heat_plot(
            samples_model[:, 0].cpu(),
            samples_model[:, 1].cpu(),
            directory,
            file_name,
            lower_limit=None,
        )
        save_path = directory + "fitted_torus.png"
        plot_heatmap_on_torus(samples_model[:, 0], samples_model[:, 1] - 2, save_path)
        save_path = directory + "wrapped_normal_torus.png"
        plot_heatmap_on_torus(
            samples_wrapped[:, 0].cpu(), samples_wrapped[:, 1].cpu() - 2, save_path
        )
