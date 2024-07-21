"""Auxiliar functions for models definition

"""

import torch


def inverse_sigmoid(x):
    """inverse sigmoid function"""
    return -torch.log(1 / (x + 1e-8) - 1)


def pseudo_inverse_softmax(x):
    """pseudo inverse of the softmax function
    Args:
        x (torch.tensor): input tensor
    Returns:
        torch.tensor: output tensor
    """
    return torch.log(x)
