import numpy as np
import numpy.typing as nmt
from typing import Tuple
from linear.matrix import Matrix
from linear.svd import svd
from metrics import kernels


class SVD_PCA:
    """
    A class for performing Principal Component Analysis (PCA) using Singular Value Decomposition (SVD) with optional kernel support.
    
    This class allows for dimensionality reduction through PCA, which can be useful for data visualization, noise reduction, or preparing data for machine learning models.
    It supports several kernel functions to allow for non-linear dimensionality reduction through kernel-PCA if needed.
    
    Attributes:
        k (int | None): Number of principal components to retain. If None, all components are kept.
        kernel (str): Type of kernel to use. Options include 'linear', 'rbf', 'poly', 'laplacian', and 'sigmoid'. Defaults to 'linear'.
        pca_model (Matrix | None): The principal component matrix after fitting the model. Initialized as None.
        explained_variances (np.ndarray | None): Variance explained by each principal component, computed during the fit process.
    
    Methods:
        fit(X: Matrix) -> Tuple[Matrix, nmt.NDArray, Matrix]:
            Fits the PCA model to the data X and returns the principal components, explained variances, and transformed data.
    """
    def __init__(self, k: int | None = None, kernel: str = 'linear'):
        """
        Initializes the SVD_PCA class with specified components and kernel.

        Args:
            k (int | None, optional): Number of principal components to retain. If None, all components are retained. Defaults to None.
            kernel (str, optional): Kernel type for transforming the input data in kernel-PCA. 
                                    Options include 'linear' (default), 'rbf', 'poly', 'laplacian', and 'sigmoid'.
        """
        self.k = k
        self.kernel = kernel
        self.pca_model = None
        self.explained_variances = None

    def fit(self, X: Matrix) -> Tuple[Matrix, nmt.NDArray, Matrix]:
        """
        Fits the PCA model to the input data X and performs dimensionality reduction using SVD.
        
        Based on the selected kernel, this function can apply kernel transformations before PCA.
        Computes the principal components, explained variances, and returns transformed data.

        Args:
            X (Matrix): Input data matrix, where rows represent samples and columns represent features.
        
        Returns:
            Tuple[Matrix, nmt.NDArray, Matrix]: 
                - P (Matrix): Principal components matrix.
                - explained_variances (nmt.NDArray): Variances explained by each principal component.
                - X_transformed (Matrix): Data projected onto the principal components.
        """

        if self.kernel == 'linear':
            self.kernel = None
        elif self.kernel == 'rbf':
            self.kernel = kernels.rbf_kernel
        elif self.kernel == 'poly':
            self.kernel = kernels.poly_kernel
        elif self.kernel == 'laplacian':
            self.kernel = kernels.laplacian_kernel
        elif self.kernel == 'sigmoid':
            self.kernel = kernels.sigmoid_kernel

        if self.kernel:
            X = Matrix(self.kernel(X))
        X = X.value
        X_centered = X - np.mean(X, axis=0)
        matrix_X = Matrix(X_centered)

        if self.pca_model:
            X_transformed = X_centered @ self.pca_model
        else:
            # SVD разложение
            U, S, Vt = svd(matrix_X)
            # Матрица главных компонент (V в SVD)
            P = Matrix(Vt.T)

            # Объясняемая дисперсия
            explained_variances = (S**2) / (X_centered.shape[0] - 1)

            # Преобразованные данные (проекции на главные компоненты)
            X_transformed = X_centered @ P.value
            X_transformed = Matrix(X_transformed[:, :self.k])

        return P, explained_variances, X_transformed
