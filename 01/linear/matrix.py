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
            Q[:, i] = u[:, i] / np.linalg.norm(u[:, i])

        R = np.zeros((n, m))
        for i in range(n):
            for j in range(i, m):
                R[i, j] = A[:, j] @ Q[:, i]

        Q, R = self.__adjust_sign(Q, R)
        return Q, R

    def eigen(self, tol=1e-12, maxiter=100000):
        "Find the eigenvalues of A using QR decomposition."
        A = self.value
        A_old = np.copy(A)
        A_new = np.copy(A)

        V = np.eye(A.shape[0])

        diff = np.inf
        i = 0
        while (diff > tol) and (i < maxiter):
            A_old[:, :] = A_new
            # Wilkinson shift (using the bottom-right element of A)
            mu = A_old[-1, -1]  # Shift is the last diagonal element
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
        def calculU(M):
            print(M)
            Matrix_M = Matrix(M)
            B = Matrix_M * Matrix_M.transpose() 
            eigenvalues, eigenvectors = B.eigen()
            ncols = np.argsort(eigenvalues)[::-1] 
            
            return eigenvectors[:,ncols] 

        def calculVt(M): 
            Matrix_M = Matrix(M)
            B = Matrix_M.transpose() * Matrix_M
            eigenvalues, eigenvectors = B.eigen()
            ncols = np.argsort(eigenvalues)[::-1] 
            return eigenvectors[:,ncols].T
        
        def calculSigma(M):
            matrix_M = Matrix(M)
            if np.size((matrix_M * matrix_M.transpose()).value) > np.size((matrix_M.transpose()*matrix_M).value): 
                new_M = matrix_M.transpose()*matrix_M
            else: 
                new_M = matrix_M * matrix_M.transpose()
                
            eigenvalues, eigenvectors = new_M.eigen()
            eigenvalues = np.sqrt(eigenvalues)
            #Sorting in descending order as the svd function does 
            return eigenvalues[::-1]
        
        U = calculU(self.value) 
        Sigma = calculSigma(self.value) 
        Vt = calculVt(self.value)

        return U, Sigma, Vt

