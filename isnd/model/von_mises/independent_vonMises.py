"""Mixture of independent von Mises distributions for base line comparison
"""

import torch

from isnd.model.auxiliar import inverse_sigmoid, pseudo_inverse_softmax
import torch.distributions as dist
import math


class IndependentVonMisesMixtureModel(torch.nn.Module):
    """class implementing a mixture model independent von Mises distributions"""

    def __init__(
        self,
        n_dim: int,
        n_components: int,
        init_means=None,
        initial_weights=None,
    ):
        """
        Args:
            n_dim (int): number of torsion angles
            n_components (int): number of wrapped normal distributions to mix
            init_means (torch.tensor): initial means of the isnd components
        """
        super().__init__()
        self.n_components = n_components
        self.ndim = n_dim
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
        self.init_log_concentrations = torch.log(
            5 * torch.ones(self.n_components, self.ndim)
        )
        self.log_concentrations = torch.nn.Parameter(self.init_log_concentrations)

    def evaluate_density(self, thetas):
        """Calculate the density of the mixture model at the input thetas
        Args:
            thetas: the input tensor of torsion angles

        Returns:
            density_output: the density evaluation at thetas
        """

        self.weights = torch.nn.Softmax(dim=0)(self.weights_params)
        self.means = torch.sigmoid(self.means_params) * 2 * math.pi
        thetas_expanded = thetas.unsqueeze(1)
        concentrations = torch.exp(self.log_concentrations)
        von_mise_dist = dist.VonMises(self.means, concentrations)
        log_prob_components = von_mise_dist.log_prob(thetas_expanded)
        mixture_log_probs = log_prob_components.sum(dim=-1)
        mixture_probs = torch.exp(mixture_log_probs)
        weighted_mixture_probs = self.weights * mixture_probs
        density_output = torch.sum(weighted_mixture_probs, dim=1)
        return density_output

    def forward(self, thetas):
        density_output = self.evaluate_density(thetas)
        return density_output

    def sample_from_density(self, n_samples):
        """Sample from the density specified by the model parameters

        Args:
            n_samples: number of samples to draw

        Returns:
            permuted_samples: samples drawn from the mixture of independent von Mises
        """
        self.weights = torch.nn.Softmax(dim=0)(self.weights_params)
        self.means = torch.sigmoid(self.means_params) * 2 * math.pi
        concentrations = torch.exp(self.log_concentrations)
        component_indices = torch.multinomial(self.weights, n_samples, replacement=True)
        component_counts = torch.bincount(
            component_indices, minlength=self.weights.shape[0]
        )
        total_samples = []
        for ind, num_samples in enumerate(component_counts):
            if num_samples > 0:
                distribution = dist.VonMises(self.means[ind], concentrations[ind])
                samples = distribution.sample((num_samples,))
                total_samples.append(samples)
        total_samples_tensor = torch.vstack(total_samples)
        permuted_indices = torch.randperm(total_samples_tensor.size(0))
        permuted_samples = total_samples_tensor[permuted_indices]
        shifted_samples = permuted_samples % (2 * math.pi)
        return shifted_samples

    def log_density(self, thetas):
        """evaluate the log density function specified by model parameters

        Args:
            thetas: angles to evaluate the log density at

        Returns:
            log_density_output: evaluation of the log density at theta
        """
        log_density_output = torch.log(self.evaluate_density(thetas))
        return log_density_output
