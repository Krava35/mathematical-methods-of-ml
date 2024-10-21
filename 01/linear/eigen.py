import numpy as np
from matrix import Matrix


A = np.random.random((3, 3))

print(sorted(np.linalg.eigvals(A)))
Matrix_A = Matrix(A)

print(sorted(Matrix_A.eigenvalues()))