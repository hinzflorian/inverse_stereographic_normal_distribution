"""For given samples create contour plots of fitted density 
"""

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


def density_plot(x, y, save_path, x_label, y_label, title, bw_adjust=0.5):
    # Generate the KDE plot
    ax = sns.kdeplot(
        x=x,
        y=y,
        cmap="Reds",
        fill=True,
        bw_adjust=bw_adjust,
    )
    # Set the limits for toroidal data (0 to 2*pi)
    ax.set_xlim([0, 2 * np.pi])
    ax.set_ylim([0, 2 * np.pi])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    plt.savefig(save_path)
    plt.close()


def all_density_plots(
    samples,
    save_dir,
    title,
    file_prefix,
    bw_adjust=0.5,
):
    for i in range(samples.shape[1]):
        for j in range(samples.shape[1]):
            if j > i:
                save_path = save_dir + f"{file_prefix}_contour_{i}_{j}.png"
                x_label = r"$\alpha_{" + str(i + 1) + "}$"
                y_label = r"$\alpha_{" + str(j + 1) + "}$"
                density_plot(
                    samples[:, i],
                    samples[:, j],
                    save_path,
                    x_label,
                    y_label,
                    title,
                    bw_adjust,
                )

