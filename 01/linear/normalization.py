from typing import List
import numpy as np
import numpy.typing as npt
from .matrix import Matrix


def normalization(matrix: Matrix, degree: int = 2) -> Matrix:
    distances = []
    for i in range(matrix.shape[0]-1):
        distances.append(__normalization(matrix.value[i], degree))
    return __transformation(distances, matrix.shape[1])


def __normalization(array: npt.ArrayLike, degree: int = 2) -> np.float64:
    return (np.sum(np.abs(array) ** degree)) ** (1 / degree)


def __transformation(distances: List, n: int) -> Matrix:
    n = len(distances)

    m = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        m[i] = distances[-i:] + distances[:-i]
    return Matrix(m)
