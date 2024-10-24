import numpy as np
import numpy.typing as nmt
from typing import Tuple
from linear.matrix import Matrix
from linear.svd import svd


class SVD_PCA:
    """_summary_
    """
    def __init__(self, k: int | None = None, kernel: str = 'linear'):
        """_summary_

        Args:
            k (int | None, optional): _description_. Defaults to None.
            kernel (str, optional): _description_. Defaults to 'linear'.
        """
        self.k = k
        self.kernel = kernel

    def fit(self, X: Matrix) -> Tuple[Matrix, nmt.NDArray, Matrix]:
        """
        Fit PCA from data in X.
        Args:
            X (Matrix): _description_

        Returns:
            _type_: _description_
        """
        X = X.value
        X_centered = X - np.mean(X, axis=0)
        matrix_X = Matrix(X_centered)
        # SVD разложение
        # U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        U, S, Vt = svd(matrix_X, kernel=self.kernel)
        # Матрица главных компонент (V в SVD)
        P = Matrix(Vt.T)

        # Объясняемая дисперсия
        explained_variances = (S**2) / (X_centered.shape[0] - 1)

        # Преобразованные данные (проекции на главные компоненты)
        X_transformed = X_centered @ P.value
        X_transformed = Matrix(X_transformed[:, :self.k])

        return P.value, explained_variances, X_transformed.value
