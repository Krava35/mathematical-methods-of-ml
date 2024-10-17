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
        self.matrix = value

    def transpose(self) -> NDArray:
        """
        Transponses the matrix.

        Return:
            value (np.NDArray): Transponsed matrix.
        """

    def eigenvalues(self) -> NDArray | None:
        """
        Calculate the eigenvalues of the matrix.

        Return:
            value (np.NDArray): eigenvalues of the matrix.
        """
        pass

    def eigevectors(self) -> NDArray | None:
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