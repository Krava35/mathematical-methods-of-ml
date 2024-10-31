from typing import List
import numpy as np
import numpy.typing as npt
from linear.matrix import Matrix


def normalization(matrix: Matrix, degree: int = 2) -> Matrix:
    """
    Normalize the rows of the given matrix using a specified degree.

    Normalization is done by calculating the p-norm of each row, where p is defined by the 
    `degree` parameter. The result is a transformation of the distances computed 
    between the rows of the matrix.

    Args:
    -----
    matrix : Matrix
        The input matrix to be normalized.
    degree : int, optional
        The degree of the norm used for normalization. Default is 2, which corresponds 
        to the Euclidean norm.

    Returns:
    --------
    Matrix
        A new Matrix object representing the normalized distances.
    """
    distances = []
    for i in range(matrix.shape[0]):
        distances.append(__normalization(matrix.value[i], degree))
    return __transformation(distances, matrix.shape[1])


def __normalization(array: npt.ArrayLike, degree: int = 2) -> np.float64:
    """
    Compute the p-norm of a given array.

    The p-norm is calculated by taking the absolute values of the elements to the power of 
    the specified degree and then taking the degree-th root of the sum of these values.

    Args:
    -----
    array : npt.ArrayLike
        The input array for which the norm is to be calculated.
    degree : int, optional
        The degree of the norm. Default is 2.

    Returns:
    --------
    np.float64
        The p-norm of the array.
    """
    return (np.sum(np.abs(array) ** degree)) ** (1 / degree)


def __transformation(distances: List, n: int) -> Matrix:
    """
    Transform the list of distances into a square matrix.

    The transformation involves creating a square matrix where each row corresponds to a 
    distance from the input list, organized such that the distances are wrapped around.

    Args:
    -----
    distances : List
        A list of distances calculated from the input matrix.
    n : int
        The number of columns in the resulting matrix.

    Returns:
    --------
    Matrix
        A new Matrix object representing the transformed distances as a square matrix.
    """
    n = len(distances)
    m = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        m[i] = distances[-i:] + distances[:-i]
    return Matrix(m)
