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

    def eigenvalues(self) -> NDArray | None:
        """
        Calculate the eigenvalues of the matrix.

        Return:
            value (np.NDArray): eigenvalues of the matrix.
        """
        pass

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

