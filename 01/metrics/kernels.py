import numpy as np
from linear.matrix import Matrix
from .normalization import normalization


def rbf_kernel(
        matrix: Matrix,
        gamma: np.float64 | None = None
        ):
    if not gamma:
        gamma = 1 / matrix.shape[0]

    norm_matrix = normalization(matrix, degree=2)
    square_norm_matrix = norm_matrix.value * norm_matrix.value
    gram_matrix = matrix.transpose().value * matrix.value
    return np.exp(-gamma * square_norm_matrix - 2 * gram_matrix)


def laplacian_kernel(
        matrix: Matrix,
        gamma: np.float64 | None = None
        ):
    if not gamma:
        gamma = 1 / matrix.shape[0]

    norm_matrix = normalization(matrix, degree=1)
    return np.exp(-gamma * norm_matrix.value)


def poly_kernel(
        matrix: Matrix,
        degree: np.float64 = 3,
        gamma: np.float64 | None = 1,
        c: np.float64 | None = 1
        ):

    coeff = np.full(matrix.shape[0], c)
    gram_matrix = matrix.transpose().value * matrix.value
    return (gamma * gram_matrix + coeff) ** degree


def sigmoid_kernel(
        matrix: Matrix,
        gamma: np.float64 | None = 1,
        c: np.float64 | None = 1
        ):

    coeff = np.full(matrix.shape[0], c)
    gram_matrix = matrix.transpose().value * matrix.value
    return np.tanh(gamma * gram_matrix + coeff)
