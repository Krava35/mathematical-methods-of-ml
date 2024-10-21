import numpy as np
from typing import Dict, Self
from numpy.typing import NDArray


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
        elif isinstance(other, (int, float)):
            return Matrix(self.value * other)
    
    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return Matrix(other * self.value)
        else:
            return NotImplemented


    def transpose(self) -> NDArray:
        """
        Transponses the matrix.

        Return:
            value (np.NDArray): Transponsed matrix.
        """
        return Matrix(self.value.transpose())

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
            Q[:, i] = u[:, i] / np.linalg.norm(u[:, i])

        R = np.zeros((n, m))
        for i in range(n):
            for j in range(i, m):
                R[i, j] = A[:, j] @ Q[:, i]

        return Q, R

    def eigenvalues(self, tol=1e-12, maxiter=3000):
        "Find the eigenvalues of A using QR decomposition."
        A = self.value
        A_old = np.copy(A)
        A_new = np.copy(A)

        diff = np.inf
        i = 0
        while (diff > tol) and (i < maxiter):
            A_old[:, :] = A_new
            Matrxi_A_old = Matrix(A_old)
            Q, R = Matrxi_A_old.QR_Decomposition()
            A_new[:, :] = R @ Q
            diff = np.abs(A_new - A_old).max()
            i += 1
        eigvals = np.diag(A_new)

        return eigvals

    def eigenvectors(self) -> NDArray | None:
        """
        Calculate the eigenvectors of the matrix.

        Return:
            value (np.NDArray): eigenvectors of the matrix.
        """
        pass

    def inverse(self) -> Self | None:
        """
        Calculate an inverse matrix of the matrix.

        Return:
            Matrix (new obj of class Matrix): the inversed matrix.
        """

    def svd(self) -> Dict[str, NDArray] | None:
        """
        Calculate singular value decomposition of the matrix.

        Return:
            svd (Dict[str, NDArray]): SVD in dictionary with keys: [U, S, V],
            with matched matrix values.
        """
        pass

