from linear.matrix import Matrix
from models.pca import SVD_PCA
import numpy as np
import unittest
from sklearn.decomposition import PCA


class TestPCA(unittest.TestCase):
    
    def test_covariance_and_components(self):
        X = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [10, 11, 12]])
        
        pca = SVD_PCA()
        matrix_X = Matrix(X)
        # 1. PCA через SVD
        P_svd, explained_variances_svd, X_transformed_svd = pca.fit(matrix_X)

        # 2. PCA через sklearn
        pca_sklearn = PCA()
        X_transformed_sklearn = pca_sklearn.fit_transform(X)

        # Главные компоненты (Eigenvectors)
        P_sklearn = pca_sklearn.components_.T

        # Объясняемая дисперсия (Explained variance)
        explained_variances_sklearn = pca_sklearn.explained_variance_

        # 3. Сравнение матриц главных компонент
        print("Матрица главных компонент (через SVD):")
        print(P_svd)

        print("\nМатрица главных компонент (через sklearn):")
        print(P_sklearn)

        # 4. Сравнение объясняемой дисперсии
        print("\nОбъясняемая дисперсия (через SVD):")
        print(explained_variances_svd)

        print("\nОбъясняемая дисперсия (через sklearn):")
        print(explained_variances_sklearn)

        # 5. Сравнение преобразованных данных
        print("\nПреобразованные данные (через SVD):")
        print(X_transformed_svd)

        print("\nПреобразованные данные (через sklearn):")
        print(X_transformed_sklearn)

if __name__ == "__main__":
    unittest.main()