"""Plot the components vs kl divergence
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch


def add_scatter_graph(data, color, lines):
    """Add a scatter plot to the graph.

    Args:
        data: the datta to plot
        color: color of the graph
        lines: the line object to store the legend
    """
    num_components = data[:, 0]
    metric = data[:, 1]
    plt.scatter(
        num_components,
        metric,
        color=color,
    )
    plt.plot(num_components, metric, color=color, alpha=0.5)
    lines.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color,
            markersize=10,
        )
    )


def joint_plot_von_mises_vs_isnd(data_paths, divergence_graphics_path):
    """Create joint plots for the kl and sinkhorn divergences for both von mises and isnd.

    Args:
        data_paths: dictionary containing the components vs kl divergence and sinkhorn distance for both von mises and isnd
        divergence_graphics_path: file path to save the plot
        title: title of the plot
    """
    ylabels = ["KL divergence"]
    colors = ["blue", "red"]
    data_types = ["alanine_tetrapeptide", "chignolin_big"]
    titles = [
        "Alanine tetrapeptide",
        "Chignolin",
    ]
    models = ["isnd", "von_mises"]
    metrics = ["kl"]
    for ind, data_type in enumerate(data_types):
        for metric, ylabel in zip(metrics, ylabels):
            title = f"{titles[ind]}: {ylabel} vs components"
            plt.figure(figsize=(10, 6))
            lines = []  # for storing line objects for the legend
            for model, color in zip(models, colors):
                data = torch.load(data_paths[data_type][model][metric])
                add_scatter_graph(data, color, lines)
            current_ylim = plt.ylim()
            plt.ylim(min(0, current_ylim[0]), current_ylim[1])
            plt.title(title)
            plt.xlabel("number of components")
            plt.ylabel("divergence")
            plt.legend(lines, models)
            plt.grid(True)
            plt.savefig(divergence_graphics_path + f"{data_type}_{metric}.png", dpi=300)
            plt.close()


def plot_metrics_vs_components(
    path_kl, path_sinkhorn, divergence_graphics_path, title, single_plot=False
):
    """Create joint plots for the kl and sinkhorn divergences.

    Args:
        path_kl: path to the tensor containing the components vs kl divergence
        path_sinkhorn: path to the tensor containing the components vs sinkhorn distance
        divergence_graphics_path: file path to save the plot
        title: title of the plot
    """
    # Pair the paths together for easier iteration
    data_paths = [path_kl, path_sinkhorn]
    ylabels = ["KL Divergence", "Sinkhorn Distance"]
    colors = ["blue", "red"]

    plt.figure(figsize=(10, 6))
    lines = []  # for storing line objects for the legend
    for data_path, color in zip(data_paths, colors):
        data = torch.load(data_path)
        num_components = data[:, 0]
        metric = data[:, 1]
        plt.scatter(
            num_components,
            metric,
            color=color,
        )
        plt.plot(
            num_components,
            metric,
            color=color,
            alpha=0.5,
        )
        lines.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color,
                markersize=10,
            )
        )
        if single_plot:
            break
    current_ylim = plt.ylim()
    plt.ylim(min(0, current_ylim[0]), current_ylim[1])
    plt.title(title)
    plt.xlabel("number of components")
    plt.ylabel("divergence")
    plt.legend(lines, ylabels)
    plt.grid(True)
    plt.savefig(divergence_graphics_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    # metric estimates paths
    data_path = {
        "alanine_tetrapeptide": {
            "isnd": {
                "kl": "isnd/results/alanine_tetrapeptide/isnd_kl_estimates.pt",
            },
            "von_mises": {
                "kl": "isnd/results/alanine_tetrapeptide/von_mises_kl_estimates.pt",
            },
        },
    }
    ###############################################################################################
    # joint plots
    ###############################################################################################
    divergence_graphics_path = "images/"
    joint_plot_von_mises_vs_isnd(data_path, divergence_graphics_path)
