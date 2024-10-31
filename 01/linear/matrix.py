import numpy as np
from typing import Tuple
from numpy.typing import NDArray


class Matrix:
    """
    Class for representing matrices and performing various matrix operations, including eigenvalue and eigenvector calculations, 
    matrix inversion, QR decomposition, and matrix addition, subtraction, and multiplication.
    
    Attributes:
    ----------
    value : np.NDArray
        The numerical values of the matrix as a NumPy array.
    shape : Tuple[int, int]
        The shape (rows, columns) of the matrix.

    Methods:
    ----------
    eigenvalues() -> np.NDArray | None
        Calculate the eigenvalues of the matrix.

    eigenvectors() -> np.NDArray | None
        Calculate the eigenvectors of the matrix.

    inverse() -> np.NDArray | None
        Calculate the inverse of the matrix.

    QR_Decomposition() -> Tuple[np.NDArray, np.NDArray]
        Perform QR decomposition of the matrix.

    eigen(tol=1e-10, maxiter=1000) -> Tuple[np.NDArray, np.NDArray]
        Compute the eigenvalues and eigenvectors of the matrix using the QR algorithm.
    """

    def __init__(self, value: NDArray):
        """
        Initialize the Matrix class with a given array of values.

        Args:
            value (np.NDArray): The matrix values as a NumPy array.
        
        Raises:
            TypeError: If the provided value is not a NumPy array.
        """
        if not isinstance(value, np.ndarray):
            raise TypeError("Data must be a numpy ndarray.")
        self.value = value
        self.shape = value.shape

    def __add__(self, matrix: 'Matrix') -> 'Matrix':
        """
        Adds two matrices of the same shape.

        Args:
            matrix (Matrix): Matrix to add.
        
        Returns:
            Matrix: Resulting matrix after addition.
        
        Raises:
            ValueError: If matrices have mismatched shapes.
        """
        if self.shape != matrix.shape:
            raise ValueError(f"Can't add matrix with shapes: {self.shape}, {matrix.shape}")
        return Matrix(matrix.value + self.value)

    def __sub__(self, matrix: 'Matrix') -> 'Matrix':
        """
        Subtracts one matrix from another.

        Args:
            matrix (Matrix): Matrix to subtract.
        
        Returns:
            Matrix: Resulting matrix after subtraction.
        
        Raises:
            ValueError: If matrices have mismatched shapes.
        """
        if self.shape != matrix.shape:
            raise ValueError(f"Can't substruct matrix with shapes: {self.shape}, {matrix.shape}")
        return Matrix(self.value - matrix.value)

    def __mul__(self, other: 'Matrix') -> 'Matrix':
        """
        Multiplies the matrix with another matrix or a scalar.

        Args:
            other (Matrix | int | float): Matrix or scalar to multiply with.
        
        Returns:
            Matrix: Resulting matrix after multiplication.
        
        Raises:
            ValueError: If matrices cannot be multiplied due to shape mismatch.
        """
        if isinstance(other, Matrix):
            if self.shape[1] != other.shape[0]:
                raise ValueError(f"Can't multiply matrix with shapes: {self.shape}, {other.shape}")
            return Matrix(self.value.dot((other.value)))
        if isinstance(other, (int, float)):
            return Matrix(self.value * other)

    def __rmul__(self, other: 'Matrix') -> 'Matrix':
        """
        Handles scalar or matrix multiplication from the left.

        Args:
            other (Matrix | int | float): Matrix or scalar to multiply from the left.
        
        Returns:
            Matrix: Resulting matrix after multiplication.
        
        Raises:
            ValueError: If matrices cannot be multiplied due to shape mismatch.
        """
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
        Returns the transpose of the matrix.

        Returns:
            Matrix: Transposed matrix.
        """
        return Matrix(self.value.transpose())

    def __diag_sign(self, A) -> np.NDArray:
        """
        Internal method to compute signs of the diagonal elements of a matrix.

        Args:
            A (np.NDArray): Input matrix.
        
        Returns:
            np.NDArray: Matrix with the signs of the diagonal elements of A.
        """
        D = np.diag(np.sign(np.diag(A)))
        return D

    def __adjust_sign(self, Q, R) -> Tuple[np.NDArray, np.NDArray]:
        """
        Adjusts the signs of the columns in Q and rows in R to enforce a positive diagonal in Q.

        Args:
            Q (np.NDArray): Orthogonal matrix from QR decomposition.
            R (np.NDArray): Upper triangular matrix from QR decomposition.
        
        Returns:
            Tuple[np.NDArray, np.NDArray]: Adjusted matrices Q and R.
        """
        D = self.__diag_sign(Q)

        Q[:, :] = Q @ D
        R[:, :] = D @ R

        return Q, R

    def QR_Decomposition(self) -> Tuple[np.NDArray, np.NDArray]:
        """
        Performs QR decomposition using the Gram-Schmidt process.
        
        Returns:
            Tuple[np.NDArray, np.NDArray]: Matrices Q and R from the QR decomposition.
        """
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

    def eigen(self, tol=1e-10, maxiter=1000) -> Tuple[np.NDArray, np.NDArray]:
        """
        Computes the eigenvalues and eigenvectors of the matrix using the QR algorithm.
        
        Args:
            tol (float): Tolerance for convergence of the eigenvalue approximation.
            maxiter (int): Maximum number of iterations.
        
        Returns:
            Tuple[np.NDArray, np.NDArray]: Array of eigenvalues and matrix of eigenvectors.
        """

        # def improved_wilkinson_shift(A):
        #     """Compute an improved Wilkinson shift using the last 2x2 block of A."""
        #     # Extract the last 2x2 block
        #     a = A[-2, -2]
        #     b = A[-1, -2]
        #     c = A[-1, -1]
            
        #     # Compute the eigenvalues of the 2x2 block
        #     delta = (a - c) / 2
        #     sign = np.sign(delta) if delta != 0 else 1  # Handle division by zero
        #     mu_1 = c - sign * b**2 / (np.abs(delta) + np.sqrt(delta**2 + b**2))
            
        #     # mu_1 is the Wilkinson shift we will use
        #     return mu_1
        
        A = self.value
        A_old = np.copy(A)
        A_new = np.copy(A)

        V = np.eye(A.shape[0])

        diff = np.inf
        i = 0
        while (diff > tol) and (i < maxiter):
            A_old[:, :] = A_new
            # Wilkinson Shift: Use the last two diagonal elements for better accuracy
            # mu = improved_wilkinson_shift(A_old)
            mu = 0.0000001

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
