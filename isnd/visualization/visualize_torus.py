""" Distribution visualization on the torus
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def plot_heatmap_on_torus(theta_samples, phi_samples, save_path, title=None):
    theta_samples = theta_samples % (2 * np.pi)
    phi_samples = phi_samples % (2 * np.pi)
    r, R = 0.55, 1.0  # Small and large radius of the torus
    n_bins = 600  # Number of bins for discretization
    # Discretize sample space
    theta_bins = np.linspace(0, 2 * np.pi, n_bins)
    phi_bins = np.linspace(0, 2 * np.pi, n_bins)
    heatmap, _, _ = np.histogram2d(
        theta_samples, phi_samples, bins=(theta_bins, phi_bins)
    )

    # Normalize heatmap
    heatmap_normalized = heatmap / np.max(heatmap)
    # Create torus surface for plotting
    angle = np.linspace(0, 2 * np.pi, n_bins)
    theta_grid, phi_grid = np.meshgrid(angle, angle)
    X = (R + r * np.cos(phi_grid)) * np.cos(theta_grid)
    Y = (R + r * np.cos(phi_grid)) * np.sin(theta_grid)
    Z = r * np.sin(phi_grid)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_xlim3d(-1.0, 1.0)
    ax.set_ylim3d(-1.0, 1.0)
    ax.set_zlim3d(-1.0, 1.0)
    if title is not None:
        ax.set_title(title)
    ax.set_axis_off()
    # Color mapping: map the heatmap density to a colormap
    colors = cm.jet(heatmap_normalized)
    # Plot the torus surface with heatmap
    shade = True
    ax.plot_surface(
        X,
        Y,
        Z,
        facecolors=colors,
        rstride=2,
        cstride=2,
        alpha=1.0,
        antialiased=False,
        shade=shade,
    )
    ax.set_axis_off()
    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
