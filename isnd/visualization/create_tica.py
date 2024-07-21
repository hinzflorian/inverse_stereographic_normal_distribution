"""Create Tica plot from given data
"""

# conda env: bg
import math
import pickle
from isnd.model.von_mises.independent_vonMises import IndependentVonMisesMixtureModel
import matplotlib.pyplot as plt
import numpy as np
import torch
from deeptime.decomposition import TICA
from matplotlib.colors import LogNorm

from ..model.isnd import IsndMixtureModel
from ..utils.torus import sine_cosine_transform
from ..visualization.visualize_distribution import plot_marginals
from scipy.stats import gaussian_kde


def plot_tica_density(
    tica_projections, save_tica_plot_path, x_limits=None, y_limits=None
):
    """Create a density-based tica plot from given data

    Args:
        tica_projections: projection on tica space
        save_tica_plot_path: path to save the tica plots
        x_limits: optional, limits of x-axis
        y_limits: optional, limit of y-axis
    """
    plt.axes(facecolor="w")
    plt.grid(False)
    # Calculate the point density
    xy = np.vstack([tica_projections[:, 0], tica_projections[:, 1]])
    z = gaussian_kde(xy)(xy)
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = tica_projections[:, 0][idx], tica_projections[:, 1][idx], z[idx]
    plt.scatter(x, y, c=z, s=50, cmap="jet")
    plt.xlabel("1st tIC")
    plt.ylabel("2nd tIC")
    plt.title("tICA Density Scatter Plot")
    plt.colorbar(label="Density")
    if x_limits is not None:
        plt.xlim(x_limits)
    if y_limits is not None:
        plt.ylim(y_limits)
    plt.savefig(save_tica_plot_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_tica(tica_projections, save_tica_plot_path, x_limits=None, y_limits=None):
    """Create tica plot from given data

    Args:
        tica_projections: projection on tica space
        save_tica_plot_path: path to save the tica plots
        x_limits: optional, limits of x-axis
        y_limits: optional, limit of y-axis
    """
    plt.axes(facecolor="w")
    plt.grid(False)
    plt.hist2d(
        tica_projections[:, 0],
        tica_projections[:, 1],
        bins=100,
        cmap="jet",
        norm=LogNorm(),
    )
    plt.xlabel("1st tIC")
    plt.ylabel("2nd tIC")
    plt.title("tICA Heatmap (log color scale)")
    plt.colorbar()
    if x_limits is not None:
        plt.xlim(x_limits)
        # Setting ticks for x-axis
        # plt.xticks(np.linspace(x_limits[0], x_limits[1], num=5))
    if y_limits is not None:
        plt.ylim(y_limits)
        # Setting ticks for y-axis
        # plt.yticks(np.linspace(y_limits[0], y_limits[1], num=5))
    plt.savefig(save_tica_plot_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_tica(data, save_directory, save_tica_path, sin_cos_transform=True):
    """Create tica object (project data onto tica space)

    Args:
        data: data to create tica plot from
        save_directory: directory to save the tica object
        save_tica_path: file name
    """
    if sin_cos_transform:
        data = sine_cosine_transform(data)
    tica = TICA(lagtime=10, dim=2)
    tica.fit(data)
    # save the tica object:
    with open(save_directory + save_tica_path, "wb") as f:
        pickle.dump(tica, f)


def plot_tica_comparison(
    tica_path,
    samples1,
    samples2,
    save_tica_plot_path1,
    save_tica_plot_path2,
    sin_cos_transform=False,
):
    """Generate tica plot for two different data sets

    Args:
        tica_path: path to the tica object
        samples1: first data set
        samples2: second data set
        save_tica_plot_path1: save path for the first tica plot
        save_tica_plot_path2: save path for the second tica plot
    """
    with open(tica_path, "rb") as f:
        tica_loaded = pickle.load(f)
    if sin_cos_transform:
        tica_output1 = tica_loaded.transform(sine_cosine_transform(samples1))
        tica_output2 = tica_loaded.transform(sine_cosine_transform(samples2))
    else:
        tica_output1 = tica_loaded.transform(samples1)
        tica_output2 = tica_loaded.transform(samples2)

    x_limits = (
        min(tica_output1[:, 0].min(), tica_output2[:, 0].min()),
        max(tica_output1[:, 0].max(), tica_output2[:, 0].max()),
    )
    y_limits = (
        min(tica_output1[:, 1].min(), tica_output2[:, 1].min()),
        max(tica_output1[:, 1].max(), tica_output2[:, 1].max()),
    )
    for i, (tica_output, save_tica_plot_path) in enumerate(
        zip([tica_output1, tica_output2], [save_tica_plot_path1, save_tica_plot_path2])
    ):
        plot_tica(
            tica_output,
            save_tica_plot_path,
            x_limits=x_limits,
            y_limits=y_limits,
        )


if __name__ == "__main__":
    peptide = "alanine_tetrapeptide"
    if peptide == "alanine_tetrapeptide":
        data_path = "datasets/alaninetetrapeptide/alanine_tetrapeptide.npy"
        save_directory = f"images/"
        save_tica_path = "tica_alanine_tetrapeptide.pkl"
        model_path = "isnd/checkpoints/alanine_tetrapeptide/isnd/isnd_300_150_epochs_20240315_174157_small_eigenvalue_network_param_True_learning_rate_0.01.pth"
    samples = np.load(data_path) % (2 * math.pi)
    ##########################################################################################
    generate_tica(samples, save_directory, save_tica_path, sin_cos_transform=True)
    tica_path = save_directory + save_tica_path
    n_dim = samples.shape[1]
    n_components = 300
    n_epochs = 150
    model_type = "isnd"
    if model_type == "isnd":
        model = IsndMixtureModel(n_dim, n_components)
    elif model_type == "von_mises":
        model = IndependentVonMisesMixtureModel(n_dim, n_components)
    model.load_state_dict(torch.load(model_path))
    n_samples = samples.shape[0]
    with torch.no_grad():
        samples_generated = np.array(model.sample_from_density(n_samples))
    plot_marginal_distributions = False
    if plot_marginal_distributions:
        plot_marginals(
            samples_generated.T, save_directory, f"generated_samples_{n_components}.png"
        )

    plot_true_marginal_distributions = False
    if plot_true_marginal_distributions:
        plot_marginals(samples.T, save_directory, f"true_samples_{n_components}.png")
    ###################################################################################
    save_plot_path1 = save_directory + "/tica_plot.png"
    save_plot_path2 = (
        save_directory
        + f"/tica_plot_{model_type}_{n_components}_components_{n_epochs}_epochs.png"
    )
    plot_tica_comparison(
        tica_path,
        samples,
        samples_generated,
        save_plot_path1,
        save_plot_path2,
        sin_cos_transform=True,
    )
