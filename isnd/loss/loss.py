"""Loss function
"""

import torch


def ml_loss(density_output):
    """maximum likelihood loss

    Args:
        density_output : the tensor of density evaluations at give torsion angles

    Returns:
        neg_log_likelihood: the average of the negative log likelihood of the data under the current model
    """
    neg_log_likelihood = -torch.mean(torch.log(density_output))
    return neg_log_likelihood
