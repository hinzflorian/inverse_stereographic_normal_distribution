"""Plotting functions for the isnd model
"""

import math
import os
import matplotlib.pyplot as plt
import numpy as np
from ..model.isnd import sample_from_mixture_of_isnd_multi_dim


def heat_plot(x, y, directory, filename, lower_limit=None):
    plt.hist2d(np.array(x), np.array(y), bins=(100, 100), cmap=plt.cm.jet)
    if lower_limit is None:
        lower_limit = 0
    plt.xlim(lower_limit, lower_limit + 2 * math.pi)
    plt.ylim(lower_limit, lower_limit + 2 * math.pi)
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    plt.savefig(directory + filename)
    plt.close()


def plot_marginals(samples, directory, format="png", lower_limit=None):
    """plots 2d marginals of samples

    Args:
        samples: samples to plot marginals from
        directory: directory to save the plots
    """
    n_dim = samples.shape[0]
    all_combinations = [
        (x, y) for x in range(0, n_dim) for y in range(0, n_dim) if x < y
    ]
    # create directory for 2d marginals if not exists
    directory_2d = directory + "2d_marginals/"
    if not os.path.exists(directory_2d):
        os.makedirs(directory_2d)
    # plot 2d marginals
    for ind1, ind2 in all_combinations:
        filename = f"marginal_{ind1}_{ind2}.{format}"
        heat_plot(samples[ind1], samples[ind2], directory_2d, filename, lower_limit)


def plot_marginals_isnd(n_samples, weights, sigmas, mus, directory, format="png"):
    """Plot all marginals of the learned isnd model

    Args:
        n_samples : number of samples to sample
        weights: weights tensor of isnd mixture
        sigmas : covariance matrices
        mus : means of the isnd mixture

    Returns:
        _type_: _description_
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    samples = sample_from_mixture_of_isnd_multi_dim(
        n_samples, weights.detach().cpu(), sigmas.detach().cpu(), mus.detach().cpu()
    )
    plot_marginals(samples, directory, format)