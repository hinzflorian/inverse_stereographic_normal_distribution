"""
We approximate the KL divergence between two distributions p, q if only samples from p are available by the estimator (14) in "Kullback-Leibler Divergence Estimation of Continuous Distributions"
"""
import math
import time
import torch
from pykeops.torch import LazyTensor
from sklearn.neighbors import NearestNeighbors


def nearest_neighbor_distance_sklearn(data, data_neighbors=None):
    # Move the data to CPU and convert it to a NumPy array
    data_np = data.cpu().numpy()
    # Initialize the NearestNeighbors model
    if data_neighbors == None:
        model = NearestNeighbors(n_neighbors=2, algorithm="auto", metric="euclidean")
        model.fit(data_np)
        # Find the nearest neighbors
        distances, _ = model.kneighbors(data_np)
        # Remove self-distance
        min_distances = distances[:, 1]
    else:
        model = NearestNeighbors(n_neighbors=1, algorithm="auto", metric="euclidean")
        data_neighbors_np = data_neighbors.cpu().numpy()
        model.fit(data_neighbors_np)
        # Find the nearest neighbors
        distances, indices = model.kneighbors(data_np)
        # Remove self-distance
        min_distances = distances[:, 0]
    # nearest_neighbors = indices[:, 1:]
    return torch.tensor(min_distances)


def nearest_neighbors_circ(data):
    x_i = LazyTensor(data[:, None, :])  # (N, 1, D) LazyTensor
    x_j = LazyTensor(data[None, :, :])  # (1, N, D) LazyTensor
    abs_diff = (x_i - x_j).abs()
    abs_diff_alternative = 2 * math.pi - abs_diff
    Differences_ij = (abs_diff_alternative - abs_diff).ifelse(
        abs_diff, abs_diff_alternative
    )
    Distances_ij = Differences_ij.norm2()
    distances, indices = Distances_ij.Kmin_argKmin(K=2, dim=1)
    return distances[:, 1], indices[:, 1]


def estimate_kl_density_density(samples_p, density_p, density_q):
    """Estimate KL[p||q] if p, q are known

    Args:
        samples_p: samples from distribution p
        density_p: density p
        density_q: density q
    Returns:
        kl_estimator: estimate of KL[p||q]
    """
    device = samples_p.device
    log_density_output_p = torch.log(density_p(samples_p)).to(device)
    log_density_output_q = torch.log(density_q(samples_p)).to(device)
    kl_estimator = torch.mean(log_density_output_p - log_density_output_q)
    return kl_estimator


def estimate_kl_samples_density(samples_p, log_density_q_output):
    """KL estimator for KL[p||q] if only samples from p are available
        based on the estimator (14) in "Kullback-Leibler Divergence Estimation of Continuous Distributions"
    Args:
        samples_p: samples from distribution p
        log_density_q: log density of distribution q

    Returns:
        kl_estimation: estimation of KL[p||q]
    """
    # euler mascheroni constant corresponds to \Gamma'(1):
    euler_mascheroni_constant = 0.577215664901532
    # calculate nearest neighbor distance
    start_time = time.time()
    nearest_neighbor_dist, _ = nearest_neighbors_circ(samples_p)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    n_samples = samples_p.shape[0]
    n_dim = samples_p.shape[1]
    coeff = math.gamma(n_dim / 2 + 1) / ((n_samples - 1) * math.pi ** (n_dim / 2))
    log_prob_density_distr1 = (
        math.log(coeff) - torch.log(torch.tensor(nearest_neighbor_dist)) * n_dim
    )
    device = samples_p.device
    log_prob_density_distr1 = log_prob_density_distr1.to(device)
    kl_estimator = (
        torch.mean(log_prob_density_distr1 - log_density_q_output)
        - euler_mascheroni_constant
    )
    return kl_estimator


def estimate_kl_samples_samples(samples_p, samples_q):
    """KL estimator for KL[p||q] if only samples from p and q are available
        based on the estimator (14) in "Kullback-Leibler Divergence Estimation of Continuous Distributions"

    Args:
        samples_p : samples from distribution p
        samples_q : samples from distribution q

    Returns:
        kl_estimation: estimation of KL[p||q]
    """
    n_dim = samples_p.shape[1]
    nearest_neighbor_dist_p = nearest_neighbor_distance_sklearn(samples_p)
    nearest_neighbor_dist_q = nearest_neighbor_distance_sklearn(samples_p, samples_q)
    n_samples = samples_p.shape[0]
    log_nearest_neighbor_dist_p = torch.log(torch.tensor(nearest_neighbor_dist_p))
    log_nearest_neighbor_dist_q = torch.log(torch.tensor(nearest_neighbor_dist_q))
    kl_estimator = n_dim * torch.mean(
        log_nearest_neighbor_dist_q - log_nearest_neighbor_dist_p
    ) + math.log(n_samples / (n_samples - 1))
    return kl_estimator
