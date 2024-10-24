import numpy as np
from metrics import kernels
from .matrix import Matrix


def svd(matrix: Matrix, kernel: str = 'linear'):

    if kernel == 'linear':
        kernel = None
    elif kernel == 'rbf':
        kernel = kernels.rbf_kernel
    elif kernel == 'poly':
        kernel = kernels.poly_kernel
    elif kernel == 'laplacian':
        kernel = kernels.laplacian_kernel
    elif kernel == 'sigmoid':
        kernel = kernels.sigmoid_kernel

    def calculU(M):
        matrix_M = Matrix(M)

        if kernel:
            B = kernel(matrix_M.transpose())
        else:
            B = matrix_M * matrix_M.transpose()  # B = M M^T

        eigenvalues, eigenvectors = B.eigen()  # Eigenvectors are left singular vectors

        # Sort eigenvalues and corresponding eigenvectors
        sorted_indices = np.argsort(eigenvalues)[::-1]
        U = eigenvectors[:, sorted_indices]  # Rearrange eigenvectors in descending order
        return U

    def calculVt(M):
        matrix_M = Matrix(M)

        if kernel:
            B = kernel(matrix_M)
        else:
            B = matrix_M.transpose() * matrix_M  # B = M M^T

        eigenvalues, eigenvectors = B.eigen()  # Eigenvectors are right singular vectors (V)

        # Sort eigenvalues and corresponding eigenvectors
        sorted_indices = np.argsort(eigenvalues)[::-1]
        Vt = eigenvectors[:, sorted_indices].T  # Transpose V to get V^T
        return Vt

    def calculSigma(M):
        matrix_M = Matrix(M)

        # Compute eigenvalues of M M^T or M^T M
        if np.size((matrix_M * matrix_M.transpose()).value) > np.size((matrix_M.transpose() * matrix_M).value):
            if kernel:
                new_M = kernel(matrix_M)
            else:
                new_M = matrix_M.transpose() * matrix_M
        else:
            if kernel:
                new_M = kernel(matrix_M.transpose())
            else:
                new_M = matrix_M * matrix_M.transpose()       

        eigenvalues, _ = new_M.eigen()

        # Singular values are the square roots of the eigenvalues
        singular_values = np.sqrt(np.abs(eigenvalues))  # Ensure no negative values due to numerical issues

        # Sorting in descending order
        sorted_singular_values = np.sort(singular_values)[::-1]
        return sorted_singular_values

    U = calculU(matrix.value) 
    Sigma = calculSigma(matrix.value)
    Vt = calculVt(matrix.value)

    return U, Sigma, Vt
