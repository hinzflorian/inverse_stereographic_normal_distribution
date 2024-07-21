"""Cluster data on torus
"""
import torch
from sklearn.cluster import KMeans

from ..utils.torus import embedding_2d_space, transform_cartesian_to_angular


def cluster_on_torus(samples, num_cluster):
    """cluster samples on the torus

    Args:
        samples: samples on the torus
        num_cluster: number of clusters

    Returns:
        angle_centers: cluster centers on the torus
    """
    samples_embedded = embedding_2d_space(samples)
    kmeans = KMeans(n_clusters=num_cluster)
    kmeans.fit(samples_embedded.detach().cpu().numpy())
    centers = kmeans.cluster_centers_
    angle_centers = transform_cartesian_to_angular(torch.tensor(centers))
    return angle_centers
