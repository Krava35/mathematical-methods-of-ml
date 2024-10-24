import numpy as np
from linear.matrix import Matrix


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
