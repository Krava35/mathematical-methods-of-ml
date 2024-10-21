from typing import List
import numpy as np
import numpy.typing as npt
from .matrix import Matrix


def normalization(matrix: Matrix, degree: int = 2) -> Matrix:
    distances = []
    for i in range(matrix.shape[0]-1):
        for j in range(matrix.shape[1]-i-1):
            distances.append(__normalization(matrix.value[i], matrix.value[j+i+1], degree))
    return __transformation(distances, matrix.shape[1])


def __normalization(array1: npt.ArrayLike, array2: npt.ArrayLike, degree: int = 2) -> np.float64:
    return (np.sum((array1 - array2) ** degree)) ** (1 / degree)


def __transformation(distances: List, n: int) -> Matrix:
    n = len(distances)

    if n == 1:
        return Matrix(np.array(distances))

    m = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i != j:
                m[i, j] = distances[i] if i < j else distances[j]
    return Matrix(m)
