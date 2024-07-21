"""Visualize results for isnd for varying variances.
"""

from isnd.model.isnd import isnd
import matplotlib.pyplot as plt
import torch
import math


def visualize_modality(variance_list):
    """Create a joint plot showing the densities for different variances.

    Args:
        variance_list (list): List of variances to plot densities for.
    """
    thetas = (
        torch.linspace(-math.pi, math.pi, 1000).reshape(-1, 1).to(dtype=torch.float64)
    )
    plt.figure(figsize=(10, 6))
    for variance in variance_list:
        variance_tensor = torch.tensor([[variance]]).to(dtype=torch.float64)
        mu = torch.tensor([[0.0]]).to(dtype=torch.float64)
        density_values = isnd(thetas, variance_tensor, mu)
        thetas_np = thetas.detach().numpy()
        density_values_np = density_values.detach().numpy()
        plt.plot(thetas_np, density_values_np, label=f"$\sigma^2 = {variance}$")
    plt.xlabel("$\\alpha$")
    plt.ylabel("Density")
    plt.title("Density for varying variances")
    plt.legend()
    plt.grid(True)
    plt.savefig("isnd/images/article_experiments/densities_plot.png")
    plt.close()


if __name__ == "__main__":
    variance_list = [0.1, 0.3, 0.5, 0.7]
    visualize_modality(variance_list)
