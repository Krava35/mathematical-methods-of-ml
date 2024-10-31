import numpy as np
from numpy.typing import NDArray
from linear.matrix import Matrix
from .normalization import normalization


def rbf_kernel(
        matrix: Matrix,
        gamma: np.float64 | None = None
        ) -> NDArray:
    """
    Compute the Radial Basis Function (RBF) kernel matrix for the given matrix.

    The RBF kernel is defined as:
        K(x_i, x_j) = exp(-gamma * ||x_i - x_j||^2)

    Args:
    -----
    matrix : Matrix
        The input matrix for which the RBF kernel is to be computed.
    gamma : np.float64 | None, optional
        Parameter that defines the width of the RBF kernel. If None, defaults to 1/n, 
        where n is the number of samples in the matrix.

    Returns:
    --------
    np.ndarray
        The computed RBF kernel matrix.
    """
    if not gamma:
        gamma = 1 / matrix.shape[0]

    sq_norms = np.sum(matrix.value ** 2, axis=1).reshape(-1, 1)
    K = sq_norms + sq_norms.T - 2 * np.dot(matrix.value, matrix.transpose().value)
    return np.exp(-gamma * K)


def laplacian_kernel(
        matrix: Matrix,
        gamma: np.float64 | None = None
        ) -> NDArray:
    """
    Compute the Laplacian kernel matrix for the given matrix.

    The Laplacian kernel is defined as:
        K(x_i, x_j) = exp(-gamma * ||x_i - x_j||)

    Args:
    -----
    matrix : Matrix
        The input matrix for which the Laplacian kernel is to be computed.
    gamma : np.float64 | None, optional
        Parameter that defines the width of the Laplacian kernel. If None, defaults to 1/n, 
        where n is the number of samples in the matrix.

    Returns:
    --------
    np.ndarray
        The computed Laplacian kernel matrix.
    """
    if not gamma:
        gamma = 1 / matrix.shape[0]

    norm_matrix = normalization(matrix, degree=1)
    return np.exp(-gamma * norm_matrix.value)


def poly_kernel(
        matrix: Matrix,
        degree: np.float64 = 3,
        gamma: np.float64 | None = 1,
        c: np.float64 | None = 1
        ) -> NDArray:
    """
    Compute the polynomial kernel matrix for the given matrix.

    The polynomial kernel is defined as:
        K(x_i, x_j) = (gamma * (x_i^T * x_j) + c)^degree

    Args:
    -----
    matrix : Matrix
        The input matrix for which the polynomial kernel is to be computed.
    degree : np.float64, optional
        The degree of the polynomial kernel. Default is 3.
    gamma : np.float64 | None, optional
        Parameter that scales the inner product. Default is 1.
    c : np.float64 | None, optional
        Coefficient added to the product. Default is 1.

    Returns:
    --------
    np.ndarray
        The computed polynomial kernel matrix.
    """
    coeff = np.full(matrix.shape[0], c)
    gram_matrix = matrix.value @ matrix.transpose().value
    return (gamma * gram_matrix + coeff) ** degree


def sigmoid_kernel(
        matrix: Matrix,
        gamma: np.float64 | None = 1,
        c: np.float64 | None = 1
        ) -> NDArray:
    """
    Compute the sigmoid kernel matrix for the given matrix.

    The sigmoid kernel is defined as:
        K(x_i, x_j) = tanh(gamma * (x_i^T * x_j) + c)

    Args:
    -----
    matrix : Matrix
        The input matrix for which the sigmoid kernel is to be computed.
    gamma : np.float64 | None, optional
        Parameter that scales the inner product. Default is 1.
    c : np.float64 | None, optional
        Coefficient added to the product. Default is 1.

    Returns:
    --------
    np.ndarray
        The computed sigmoid kernel matrix.
    """
    coeff = np.full(matrix.shape[0], c)
    gram_matrix = matrix.value @ matrix.transpose().value
    return np.tanh(gamma * gram_matrix + coeff)
