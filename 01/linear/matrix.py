import numpy as np
from numpy.typing import NDArray
import metrics.kernels as kernels


class Matrix:
    """
    Class Matrix is used for matrix operations and getting characteristics of it.

    Attributes:
    ----------
    matrix : np.NDArray
        Matrix

    Params:
    ----------
    value : (np.NDArray)
        Matrix value.

    Methods:
    ----------
    eigenvalues() -> np.NDArray | None
        Calculate the eigenvalues of the matrix.

    eigevectors() -> np.NDArray | None
        Calculate the eigenvectors of the matrix.

    inverse() -> np.NDArray | None
        Calculate inverse matrix.
    """

    def __init__(self, value: NDArray):
        """
        Initializes the matrix with given value.

        Return:
            value (np.NDArray): Matrix value.
        """
        if not isinstance(value, np.ndarray):
            raise TypeError("Data must be a numpy ndarray.")
        self.value = value
        self.shape = value.shape

    def __add__(self, matrix: 'Matrix'):
        if self.shape != matrix.shape:
            raise ValueError(f"Can't add matrix with shapes: {self.shape}, {matrix.shape}")
        return Matrix(matrix.value + self.value)

    def __sub__(self, matrix: 'Matrix'):
        if self.shape != matrix.shape:
            raise ValueError(f"Can't substruct matrix with shapes: {self.shape}, {matrix.shape}")
        return Matrix(self.value - matrix.value)

    def __mul__(self, other: 'Matrix'):
        if isinstance(other, Matrix):
            if self.shape[1] != other.shape[0]:
                raise ValueError(f"Can't multiply matrix with shapes: {self.shape}, {other.shape}")
            return Matrix(self.value.dot((other.value)))
        if isinstance(other, (int, float)):
            return Matrix(self.value * other)

    def __rmul__(self, other: 'Matrix'):
        if isinstance(other, (int, float)):
            return Matrix(other * self.value)
        if isinstance(other, Matrix):
            if self.shape[1] != other.shape[0]:
                raise ValueError(f"Can't multiply matrix with shapes: {self.shape}, {other.shape}")
            return Matrix(np.dot(other.value, self.value))
        else:
            return NotImplemented


    def transpose(self) -> NDArray:
        """
        Transponses the matrix.

        Return:
            value (np.NDArray): Transponsed matrix.
        """
        return Matrix(self.value.transpose())

    def __diag_sign(self, A):
        "Compute the signs of the diagonal of matrix A"
        D = np.diag(np.sign(np.diag(A)))
        return D

    def __adjust_sign(self, Q, R):
        """
        Adjust the signs of the columns in Q and rows in R to
        impose positive diagonal of Q
        """

        D = self.__diag_sign(Q)

        Q[:, :] = Q @ D
        R[:, :] = D @ R

        return Q, R

    def QR_Decomposition(self):
        A = self.value
        n, m = A.shape

        Q = np.empty((n, n))
        u = np.empty((n, n))

        u[:, 0] = A[:, 0]
        Q[:, 0] = u[:, 0] / np.linalg.norm(u[:, 0])

        for i in range(1, n):

            u[:, i] = A[:, i]
            for j in range(i):
                u[:, i] -= (A[:, i] @ Q[:, j]) * Q[:, j]
            norm_u_i = np.linalg.norm(u[:, i])
            if norm_u_i > 1e-15:  # Use a small threshold to avoid division by zero
                Q[:, i] = u[:, i] / norm_u_i
            else:
                Q[:, i] = np.zeros_like(u[:, i])  # Handle zero norm gracefully

        R = np.zeros((n, m))
        for i in range(n):
            for j in range(i, m):
                R[i, j] = A[:, j] @ Q[:, i]

        Q, R = self.__adjust_sign(Q, R)
        return Q, R

    def eigen(self, tol=1e-6, maxiter=10000):
        "Find the eigenvalues of A using QR decomposition."

        def improved_wilkinson_shift(A):
            """Compute an improved Wilkinson shift using the last 2x2 block of A."""
            # Extract the last 2x2 block
            a = A[-2, -2]
            b = A[-1, -2]
            c = A[-1, -1]
            
            # Compute the eigenvalues of the 2x2 block
            delta = (a - c) / 2
            sign = np.sign(delta) if delta != 0 else 1  # Handle division by zero
            mu_1 = c - sign * b**2 / (np.abs(delta) + np.sqrt(delta**2 + b**2))
            
            # mu_1 is the Wilkinson shift we will use
            return mu_1
        A = self.value
        A_old = np.copy(A)
        A_new = np.copy(A)

        V = np.eye(A.shape[0])

        diff = np.inf
        i = 0
        while (diff > tol) and (i < maxiter):
            A_old[:, :] = A_new
            # Wilkinson Shift: Use the last two diagonal elements for better accuracy
            mu = improved_wilkinson_shift(A_old)

            A_shifted = A_old - mu * np.eye(A.shape[0])  # A - μI
            
            # Perform QR decomposition on the shifted matrix
            Matrix_A_shifted = Matrix(A_shifted)
            Q, R = Matrix_A_shifted.QR_Decomposition()

            # Update the matrix: A_new = R @ Q + μI
            A_new[:, :] = R @ Q + mu * np.eye(A.shape[0])
            V = V @ Q
            diff = np.abs(A_new - A_old).max()
            i += 1
        
        eigvals = np.diag(A_new)
        eigvecs = V

        return eigvals, eigvecs


    def svd(self, kernel: str = 'linear'):

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

        U = calculU(self.value) 
        Sigma = calculSigma(self.value)
        Vt = calculVt(self.value)

        return U, Sigma, Vt

