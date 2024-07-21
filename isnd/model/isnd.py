""" Implementation of the Inverse Stereographic
    Normal Distribution (see e.g. "On geometric probability distributions
on the torus with applications to
molecular biology" Definition 1.1.)
    """

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.nn.functional as F

from isnd.model.auxiliar import inverse_sigmoid, pseudo_inverse_softmax
from isnd.model.spd_parametrization.generate_spd import (
    generate_spd_matrices,
)


def inverse_stereographic_projection(x):
    """Inverse stereographic projections: cartesian coordinates to angles
    Args:
        x: cartesian coordinates
    Returns:
        theta: angles
    """
    theta = 2 * torch.arctan(x)
    return theta


def stereographic_projection(theta):
    """Stereographic projections: angles to cartesian coordinates
    Args:
        theta: angles
    Returns:
        x: cartesian coordinates
    """
    x = torch.sin(theta) / (1 + torch.cos(theta))
    return x


def sample_from_mixture_of_isnd_multi_dim(n_samples, weights, sigmas, mus):
    """Sample from a mixture of inverse stereographic normal distributions

    Args:
        n_samples: number of samples to generate
        weights: the weights of the mixture
        sigmas: the covariance matrices of the mixture
        mus: the means of the mixture

    Returns:
        samples: samples from the mixture of inverse stereographic normal distributions
    """
    device = sigmas.device
    samples = torch.multinomial(weights, n_samples, replacement=True)
    samples_per_normal = torch.bincount(samples)

    normal_samples = torch.randn(
        (torch.sum(samples_per_normal), mus.shape[1]),
        device=device,
        dtype=torch.float64,
    )
    lower_triangular = torch.linalg.cholesky(sigmas)
    normal_data = torch.zeros_like(normal_samples, dtype=torch.float64)
    index_thresholds = torch.cat(
        (torch.tensor([0], device=device), torch.cumsum(samples_per_normal, dim=0))
    )
    for ind in range(0, len(index_thresholds) - 1):
        index_start = index_thresholds[ind]
        index_end = index_thresholds[ind + 1]
        normal_data[index_start:index_end] = (
            lower_triangular[ind] @ normal_samples[index_start:index_end, :].t()
        ).t()
    samples_transformed = inverse_stereographic_projection(normal_data)
    for ind in range(0, len(index_thresholds) - 1):
        index_start = index_thresholds[ind]
        index_end = index_thresholds[ind + 1]
        samples_transformed[index_start:index_end] += mus[ind]
    samples = samples_transformed % (2 * math.pi)
    perm_indices = torch.randperm(samples.shape[0])
    shuffled_sampels = samples[perm_indices]
    return shuffled_sampels.t()



def isnd(theta, sigma, mu):
    """Inverse stereographic normal distribution, see Definition 1.2

    Args:
        theta: angles theta_{1},...,theta_{n} \in ]-pi,pi[
        sigma: covariance matrix
        mu: means m_{1},...,m_{n}
    Returns:
        density_output: isnd(theta) for given sigma, mu
    """
    n_dim = theta.shape[1]
    det = torch.det(sigma)

    theta_shifted = (theta - mu) % (2 * math.pi)
    theta_shifted[torch.abs(theta_shifted - math.pi) < 0.001] = 3.14

    exp_argument = (torch.sin(theta_shifted) / (torch.cos(theta_shifted) + 1)).t()
    sigma_inverse = torch.inverse(sigma)
    density_output = (
        (2 * math.pi) ** (-n_dim / 2)
        * torch.abs(det) ** (-0.5)
        * (
            torch.prod(1 / (torch.cos(theta_shifted) + 1), dim=1)
            * torch.exp(
                -1
                / 2
                * torch.einsum(
                    "ij,ij->j",
                    exp_argument,
                    sigma_inverse @ exp_argument,
                )
            )
        )
    )
    if torch.any(torch.isnan(density_output)):
        print("error: nan in density evaluation")
    return density_output


def isnd_mixture(theta, weights, sigmas, mus):
    """mixture of inverse stereographic normal distributions

    Args:
        theta : torsion angles to evaluate the density at
        weights : weights (need to sum to 1) determine the contribution of each single isnd
        sigmas : covariance matrices
        mus : vectors of means
    Returns:
        isnd_mixture_output: isnd_mixture density evaluations at theta, for given weights, sigmas, mus
    """
    isnd_mixture_output = 0
    for ind in range(0, len(weights)):
        isnd_mixture_output += weights[ind] * isnd(
            theta, sigmas[ind, :, :], mus[ind, :]
        )
    isnd_mixture_output_clamped = torch.clamp(isnd_mixture_output, 1e-40)
    return isnd_mixture_output_clamped


class ConvNet(nn.Module):
    def __init__(self, input_dim, input_dim2):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(16 * input_dim * input_dim2, 64)
        self.fc2 = nn.Linear(64, input_dim * input_dim2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class IsndMixtureModel(torch.nn.Module):
    """class implementing a model for the isnd distribution"""

    def __init__(
        self,
        n_dim: int,
        n_components: int,
        init_means=None,
        initial_weights=None,
        initial_eigenvalues=None,
        network_paramterization=True,
        initial_upper_triangular=None,
        orth_parametrization="matrix_exp",
        eigenvalue_limit=0.5,
    ):
        """
        Args:
            n_dim (int): number of torsion angles
            n_components (int): number of distributions to mix
            init_means (torch.tensor): initial means of the isnd components
            initial_eigenvalues: initial values for eigenvalues
            network_paramterization: wheather to use a network to parameterize the skew symmetric matrix
            initial_upper_triangular: initial value for upper triangular matrix (which is used to create upper triangular matrix) 
            orth_parametrization: type of parametrization of orthogonal matrices
        """
        super().__init__()
        self.network_paramterization = network_paramterization
        self.orth_parametrization = orth_parametrization
        self.n_components = n_components
        self.ndim = n_dim
        self.eigenvalue_limit = eigenvalue_limit

        if initial_weights is None:
            initial_weights = (
                torch.ones(
                    n_components,
                )
                / n_components
            )

        self.weights_params = torch.nn.Parameter(
            pseudo_inverse_softmax(initial_weights)
        )
        self.weights = torch.nn.Softmax(dim=0)(self.weights_params)
        if init_means is None:
            init_means = torch.rand(self.n_components, self.ndim) * 2 * math.pi
        self.means_params = torch.nn.Parameter(
            inverse_sigmoid(init_means / (2 * math.pi))
        )
        self.means = torch.sigmoid(self.means_params) * 2 * math.pi
        if initial_upper_triangular is None:
            initial_upper_triangular = (
                torch.randn(
                    n_components,
                    int((n_dim**2 - n_dim) / 2 + n_dim),
                    dtype=torch.float64,
                )
                * 2
            )

        if network_paramterization:
            num_triangular_entries = (n_dim**2 - n_dim) / 2 + n_dim
            self.layers = ConvNet(
                self.n_components, int(num_triangular_entries)
            ).double()

            self.upper_triangle = torch.nn.Parameter(initial_upper_triangular).double()
            output = self.layers(self.upper_triangle.reshape(1, 1, n_components, -1))
            self.skew_symmetric_matrices_upper_triangular = output.reshape(
                self.n_components, -1
            )
        else:
            self.skew_symmetric_matrices_upper_triangular = torch.nn.Parameter(
                initial_upper_triangular
            )
        if initial_eigenvalues is None:
            self.eigenvalue_vector_unscaled = torch.nn.Parameter(
                inverse_sigmoid(torch.ones(self.n_components, self.ndim) * 0.2)
            )
        else:
            self.eigenvalue_vector_unscaled = torch.nn.Parameter(
                inverse_sigmoid(2 * initial_eigenvalues)
            )
        self.eigenvalue_vectors = (
            torch.sigmoid(self.eigenvalue_vector_unscaled) * self.eigenvalue_limit
        )
        self.covariance_matrices = generate_spd_matrices(
            self.eigenvalue_vectors,
            self.skew_symmetric_matrices_upper_triangular.to(dtype=torch.float64),
            self.orth_parametrization,
        )

    def updata_parameters(self):
        """update the parameters of the model"""
        self.weights = torch.nn.Softmax(dim=0)(self.weights_params)
        self.means = torch.sigmoid(self.means_params) * 2 * math.pi
        self.eigenvalue_vectors = (
            torch.sigmoid(self.eigenvalue_vector_unscaled) * self.eigenvalue_limit
        )
        if self.network_paramterization:
            output = self.layers(
                self.upper_triangle.reshape(1, 1, self.n_components, -1)
            )
            self.skew_symmetric_matrices_upper_triangular = output.reshape(
                self.n_components, -1
            )

        self.covariance_matrices = generate_spd_matrices(
            self.eigenvalue_vectors,
            self.skew_symmetric_matrices_upper_triangular,
            self.orth_parametrization,
        )

    def forward(self, thetas):
        """Forward function of the network
        Args:
            thetas: the input tensor of torsion angles

        Returns:
            density_output: the output of the model specifiying the denstity evaluation at theta
        """
        self.updata_parameters()
        density_output = isnd_mixture(
            thetas,
            self.weights,
            self.covariance_matrices,
            self.means,
        )
        return density_output

    def evaluate_density(self, thetas):
        """evaluate the density function specified by model parameters

        Args:
            thetas: angles to evaluate the density at

        Returns:
            density_output: evaluation of the density at theta
        """
        device = self.weights_params.device
        self.updata_parameters()
        density_output = isnd_mixture(
            thetas.to(device),
            self.weights,
            self.covariance_matrices,
            self.means,
        )
        return density_output

    def sample_from_density(self, n_samples):
        """sample from the density function specified by model parameters

        Args:
            n_samples : number of samples to draw

        Returns:
            samples: samples drawn from the density function
        """
        self.updata_parameters()
        samples = sample_from_mixture_of_isnd_multi_dim(
            n_samples,
            self.weights,
            self.covariance_matrices,
            self.means,
        ).t()
        return samples

    def log_density(self, thetas):
        """evaluate the log density function specified by model parameters

        Args:
            thetas: angles to evaluate the log density at

        Returns:
            log_density_output: evaluation of the log density at theta
        """
        self.updata_parameters()
        log_density_output = torch.log(self.evaluate_density(thetas))
        return log_density_output
