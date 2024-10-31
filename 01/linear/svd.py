import numpy as np
from typing import Tuple
from numpy.typing import NDArray
from .matrix import Matrix


def svd(matrix: Matrix) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Perform Singular Value Decomposition (SVD) on the given matrix.

    The SVD decomposes the input matrix into three matrices U, Sigma, and Vt such that:
        matrix ≈ U * Sigma * Vt

    Args:
    -----
    matrix : Matrix
        The input matrix to be decomposed.

    Returns:
    --------
    Tuple[np.NDArray, np.NDArray, np.NDArray]
        A tuple containing:
        - U (np.NDArray): The left singular vectors of the matrix.
        - Sigma (np.NDArray): The singular values of the matrix, sorted in descending order.
        - Vt (np.NDArray): The transpose of the right singular vectors of the matrix.

    Notes:
    ------
    - The left singular vectors (U) are calculated by finding the eigenvectors of M * M^T.
    - The right singular vectors (Vt) are calculated by finding the eigenvectors of M^T * M.
    - The singular values (Sigma) are obtained by taking the square roots of the eigenvalues from either
      M * M^T or M^T * M.
    """

    def calculU(M: NDArray | Matrix) -> NDArray:
        """
        Calculate the left singular vectors (U) of the matrix.

        Args:
            M (np.NDArray): The input matrix.

        Returns:
            np.NDArray: The left singular vectors of the matrix.
        """
        matrix_M = Matrix(M)

        B = matrix_M * matrix_M.transpose()  # B = M M^T

        eigenvalues, eigenvectors = B.eigen()  # Eigenvectors are left singular vectors

        # Sort eigenvalues and corresponding eigenvectors
        sorted_indices = np.argsort(eigenvalues)[::-1]
        U = eigenvectors[:, sorted_indices]  # Rearrange eigenvectors in descending order
        return U

    def calculVt(M: NDArray | Matrix) -> NDArray:
        """
        Calculate the transpose of the right singular vectors (V^T) of the matrix.

        Args:
            M (np.NDArray): The input matrix.

        Returns:
            np.NDArray: The transpose of the right singular vectors of the matrix.
        """
        matrix_M = Matrix(M)

        B = matrix_M.transpose() * matrix_M  # B = M M^T

        eigenvalues, eigenvectors = B.eigen()  # Eigenvectors are right singular vectors (V)

        # Sort eigenvalues and corresponding eigenvectors
        sorted_indices = np.argsort(eigenvalues)[::-1]
        Vt = eigenvectors[:, sorted_indices].T  # Transpose V to get V^T
        return Vt

    def calculSigma(M: NDArray | Matrix) -> NDArray:
        """
        Calculate the singular values (Sigma) of the matrix.

        Args:
            M (np.NDArray): The input matrix.

        Returns:
            np.NDArray: The singular values of the matrix, sorted in descending order.
        """
        matrix_M = Matrix(M)

        # Compute eigenvalues of M M^T or M^T M
        if np.size((matrix_M * matrix_M.transpose()).value) > np.size((matrix_M.transpose() * matrix_M).value):
                new_M = matrix_M.transpose() * matrix_M
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
