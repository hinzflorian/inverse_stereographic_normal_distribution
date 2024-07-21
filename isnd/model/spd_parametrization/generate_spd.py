"""Different parametrizations of SPD matrices
"""

import torch


def matrix_exponential(X):
    """Compute the matrix exponential of a matrix X"""
    return torch.matrix_exp(X)


def cayley_transform(skew_sym_matrices):
    """Compute the Cayley transformation on a batch of skew-symmetric matrices to obtain a batch of orthogonal matrices

    Args:
        skew_sym_matrices: batch of skew-symmetric matrices
    Returns:
        orthogonal_matrices: batch of orthogonal matrices
    """
    device = skew_sym_matrices.device
    id_matrices = torch.eye(skew_sym_matrices.shape[1], device=device).repeat(
        skew_sym_matrices.shape[0], 1, 1
    )
    orthogonal_matrices = (id_matrices - skew_sym_matrices) @ torch.inverse(
        id_matrices + skew_sym_matrices
    )
    return orthogonal_matrices


def cayley_transform_stabilized(skew_sym_matrices):
    """Compute the Cayley transformation on a batch of skew-symmetric matrices to obtain a batch of orthogonal matrices

    Args:
        skew_sym_matrices: batch of skew-symmetric matrices
    Returns:
        orthogonal_matrices: batch of orthogonal matrices
    """
    batch_size, n_dim, _ = skew_sym_matrices.shape
    device = skew_sym_matrices.device

    # Create the identity matrix once and expand it to the required shape
    id_matrices = (
        torch.eye(n_dim, device=device).unsqueeze(0).expand(batch_size, -1, -1)
    )

    # Use matrix solve instead of inverse
    rhs = id_matrices - skew_sym_matrices
    solution = torch.linalg.solve(rhs, id_matrices + skew_sym_matrices)

    return solution


def skew_symmetric_matrices(n, upper_triangular_entries_skew_matrices):
    """Function to create a skew-symmetric matrices from its upper triangular entries

    Args:
        n: matrix is of dimension n x n

    Returns:
        skew_sym_matrix: the skew-symmetric matrix
    """
    triangular_indexes = torch.triu_indices(n, n)
    upper_triangular_matrices = torch.zeros(
        upper_triangular_entries_skew_matrices.shape[0],
        n,
        n,
        device=upper_triangular_entries_skew_matrices.device,
        dtype=torch.float64,
    )
    upper_triangular_matrices[:, triangular_indexes[0, :], triangular_indexes[1, :]] = (
        upper_triangular_entries_skew_matrices
    )
    skew_sym_matrices = upper_triangular_matrices - upper_triangular_matrices.transpose(
        1, 2
    )
    return skew_sym_matrices


def generate_spd_matrices(
    diag_vectors, upper_triangular_entries_skew_matrices, method="matrix_exp"
):
    """generate spd matrices from a batch of diagonal vectors and skew-symmetric matrices

    Args:
        diag_vectors: n_components diagonal vectors with eigenvalues
        upper_triangular_entries_skew_matrices: matrix entries of skew symmetric matrices
    """
    n_dim = diag_vectors.shape[1]
    skew_sym_matrices = skew_symmetric_matrices(
        n_dim, upper_triangular_entries_skew_matrices
    )
    if method == "cayley":
        orthogonal_matrices = cayley_transform(skew_sym_matrices)
    elif method == "matrix_exp":
        orthogonal_matrices = matrix_exponential(skew_sym_matrices)
    diagonal_matrices = torch.diag_embed(diag_vectors).to(dtype=torch.float64)
    spd_matrices = (
        orthogonal_matrices.transpose(1, 2) @ diagonal_matrices @ orthogonal_matrices
    )
    return spd_matrices