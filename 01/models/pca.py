import numpy as np
import tqdm
from linear.matrix import Matrix
from linear.normalization import normalization


class SVD_PCA:
    def __init__(self, k: int | None = None, kernel: str = 'linear'):
        self.k  = k
        self.kernel = kernel

    def fit(self, X: Matrix):
        X = X.value
        X_centered = X - np.mean(X, axis=0)
        matrix_X = Matrix(X_centered)    
        # SVD разложение
        # U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        U, S, Vt = matrix_X.svd()
        # Матрица главных компонент (V в SVD)
        P = Vt.T
        
        # Объясняемая дисперсия
        explained_variances = (S**2) / (X_centered.shape[0] - 1)
        
        # Преобразованные данные (проекции на главные компоненты)
        X_transformed = X_centered @ P
        
        return P, explained_variances, X_transformed

    def __rbf_kernel(self, matrix: Matrix, gamma: np.float64 | None = None):
        if not gamma:
            gamma = 1 / matrix.shape[0]
        norm_matrix = normalization(matrix, degree=2)
        return np.exp(-gamma * (norm_matrix.value * norm_matrix.value - 2 * matrix.transpose().value() * matrix.value()))

    def __laplacian_kernel(self, matrix: Matrix, gamma: np.float64 | None = None):
        if not gamma:
            gamma = 1 / matrix.shape[0]
        norm_matrix = normalization(matrix, degree=1)
        return np.exp(-gamma * norm_matrix.value)

    def __poly_kernel(self, matrix: Matrix, degree: np.float64 = 3, gamma: np.float64 | None = 1, c: np.float64 | None = 1):
        coeff = np.full(matrix.shape[0], c)
        return (gamma * matrix.transpose().value() * matrix.value() + coeff) ** degree
    
    def __sigmoid_kernel(self, matrix: Matrix, gamma: np.float64 | None = 1, c: np.float64 | None = 1):
        coeff = np.full(matrix.shape[0], c)
        return np.tanh(gamma * matrix.transpose().value() * matrix.value() + coeff)
