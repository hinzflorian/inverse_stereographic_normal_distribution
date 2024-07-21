import math

import numpy as np
import torch


def embedding_2d_space(samples, r=1):
    """Embed the samples in a 2D cartesian space (where D is the dimensionality of the torus [0, 2pi]^D])
    Args:
        samples: torch.Tensor, shape (n_samples, n_dim)
    Returns:
        samples_embedded: torch.Tensor, shape (n_samples, 2*n_dim   )
    """
    samples_embedded = torch.zeros(
        samples.shape[0], 2 * samples.shape[1], device=samples.device
    )
    samples_embedded[:, 1::2] = r * torch.sin(samples)
    samples_embedded[:, ::2] = r * torch.cos(samples)
    return samples_embedded


def sine_cosine_transform(data):
    """Transform data with sine and cosine for periodic features

    Args:
        data: data to be transformed

    Returns:
        np.array: transformed data
    """
    return np.hstack((np.sin(data), np.cos(data)))


def transform_cartesian_to_angular(cartesian_coordinates):
    """Transform the 2D cartesian coordinates back to angular coordinates

    Args:
        cartesian_coordinates:  cartesian coordinates of dimension (n_samples, 2*n_dim)
    Returns:
        angles: angular representation of the cartesian coordinates
    """
    normalization_constant = torch.sqrt(
        cartesian_coordinates[:, 1::2] ** 2 + cartesian_coordinates[:, ::2] ** 2
    )
    # we need to normalize the vectors, to ensure that they have length 1 (endpoints are on the unit circle)
    asin_values = torch.arcsin(cartesian_coordinates[:, 1::2] / normalization_constant)
    acos_values = torch.arccos(cartesian_coordinates[:, ::2] / normalization_constant)
    angles = (acos_values * torch.sgn(asin_values)) % (2 * math.pi)
    return angles
