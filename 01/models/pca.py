import numpy as np
import tqdm
from linear.matrix import Matrix
from linear.normalization import normalization


class PCA:
    def __init__(self, k: int | None = None, kernel: str = 'linear'):
        self.k  = k
        self.kernel = kernel

    def fit(self) -> Matrix:
        pass

    def __rbf_kernel(self, matrix: Matrix, gamma: np.float64 | None = None):
        if not gamma:
            gamma = 1 / matrix.shape(0)
        norm_matrix = normalization(matrix, degree=2)
        return np.exp(-gamma * norm_matrix.value)

    def __laplacian_kernel(self, matrix: Matrix, gamma: np.float64 | None = None):
        if not gamma:
            gamma = 1 / matrix.shape(0)
        norm_matrix = normalization(matrix, degree=1)
        return np.exp(-gamma * norm_matrix.value)

    def __poly_kernel(self, matrix: Matrix, degree: np.float64 = 3, gamma: np.float64 | None = None, c: np.float64 | None = None):
        if not gamma:
            gamma = 1 / matrix.shape(0)
        if not c:
            c = 1
        coeff = np.full(matrix.shape, c)
        return (gamma * matrix.transpose().value() * matrix.value() + coeff) ** degree
    
    def __sigmoid_kernel(self, matrix: Matrix, gamma: np.float64 | None = None, c: np.float64 | None = None):
        if not gamma:
            gamma = 1 / matrix.shape(0)
        if not c:
            c = 1
        coeff = np.full(matrix.shape, c)
        return np.tanh(gamma * matrix.transpose().value() * matrix.value() + coeff)
